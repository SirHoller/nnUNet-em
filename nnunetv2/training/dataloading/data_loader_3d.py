import numpy as np
import torch
import logging
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset

from typing import Union, Tuple

from batchgenerators.dataloading.data_loader import DataLoader
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.utilities.label_handling.label_handling import LabelManager
logging.basicConfig(level=logging.INFO)

class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []
        # print(f"nnUNetDataLoader3D")

        for j, i in enumerate(selected_keys):
            # oversampling foreground will improve stability of model training, especially if many patches are empty
            # (Lung for example)
            force_fg = self.get_do_oversample(j)

            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # If we are doing the cascade then the segmentation from the previous stage will already have been loaded by
            # self._data.load_case(i) (see nnUNetDataset.load_case)
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

            # whoever wrote this knew what he was doing (hint: it was me). We first crop the data to the region of the
            # bbox that actually lies within the data. This will result in a smaller array which is then faster to pad.
            # valid_bbox is just the coord that lied within the data cube. It will be padded to match the patch size
            # later
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # At this point you might ask yourself why we would treat seg differently from seg_from_previous_stage.
            # Why not just concatenate them here and forget about the if statements? Well that's because segneeds to
            # be padded with -1 constant whereas seg_from_previous_stage needs to be padded with 0s (we could also
            # remove label -1 in the data augmentation but this way it is less error prone)
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)

        # Aplicar transformaciones
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


def calculate_class_distribution(dataset):
    """
    Calcula la proporción de cada clase en el conjunto de datos.
    """
    all_labels = []
    for case in dataset.keys():
        if 'label_file' in dataset[case]:
            label_file = dataset[case]['label_file']
            label_data = torch.from_numpy(np.load(label_file)).flatten()
        else:
            data_file = dataset[case]['data_file']
            with np.load(data_file) as data:
                label_data = torch.from_numpy(data['seg']).flatten()  # Las etiquetas están bajo la clave 'seg'
        all_labels.append(label_data)
    all_labels = torch.cat(all_labels)
    unique, counts = torch.unique(all_labels, return_counts=True)
    total_voxels = all_labels.numel()
    return {int(k): v.item() / total_voxels for k, v in zip(unique, counts)}


    
class nnUNetDataLoader3DMinorityClass(nnUNetDataLoaderBase):
    def __init__(self,
                 data: nnUNetDataset,
                 batch_size: int,
                 patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 final_patch_size: Union[List[int], Tuple[int, ...], np.ndarray],
                 label_manager: LabelManager,
                 oversample_foreground_percent: float = 0.0,
                 sampling_probabilities: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 pad_sides: Union[List[int], Tuple[int, ...], np.ndarray] = None,
                 probabilistic_oversampling: bool = False,
                 transforms=None):

        # Llamamos al __init__ de la clase base con los argumentos recibidos
        super().__init__(data=data,
                         batch_size=batch_size,
                         patch_size=patch_size,
                         final_patch_size=final_patch_size,
                         label_manager=label_manager,
                         oversample_foreground_percent=oversample_foreground_percent,
                         sampling_probabilities=sampling_probabilities,
                         pad_sides=pad_sides,
                         probabilistic_oversampling=probabilistic_oversampling,
                         transforms=transforms)
        self.dynamic_oversampling = True
        self.class_distribution = None
        self.dataset = self._data.dataset
        if self.dynamic_oversampling:
            self.determine_class_distribution()

    def determine_class_distribution(self):
        """
        Calcula la distribución de clases en el conjunto de entrenamiento.
        """
        self.class_distribution = calculate_class_distribution(self.dataset)
        # print(f"Class distribution: {self.class_distribution}")

    def get_least_represented_class(self):
        """
        Encuentra la clase menos representada.
        """
        if self.class_distribution is None:
            raise ValueError("Class distribution has not been calculated.")
        return min(self.class_distribution, key=self.class_distribution.get)

    def checking_least_represented_class(self, i):
        """
        Comprueba si el caso contiene la clase menos representada.
        """
        data_file = self.dataset[i]['data_file']
        with np.load(data_file) as data:
            seg = torch.from_numpy(data['seg']).flatten()
        least_represented_class = self.get_least_represented_class()
        return least_represented_class in seg
    
    
    # def get_indices(self):
    #     """
    #     Obtiene índices para el siguiente lote, priorizando ejemplos con clases menos representadas si es necesario.
    #     """
    #     if self.infinite:
    #         # Selección con oversampling dinámico si las probabilidades están definidas
    #         if self.sampling_probabilities is not None:
    #             return np.random.choice(self.indices, self.batch_size, replace=True, p=self.sampling_probabilities)
    #         else:
    #             return np.random.choice(self.indices, self.batch_size, replace=True)

    #     if self.last_reached:
    #         self.reset()
    #         raise StopIteration

    #     if not self.was_initialized:
    #         self.reset()

    #     indices = []

    #     # Dinámico: Priorizamos ejemplos con clases menos representadas
    #     if hasattr(self, 'dynamic_oversampling') and self.dynamic_oversampling:
    #         least_represented_class = self.get_least_represented_class()
    #         # Filtramos índices que contienen la clase menos representada
    #         priority_indices = [idx for idx in self.indices if self.checking_least_represented_class(idx)]
    #         if priority_indices:
    #             # Selección proporcional para priorizar casos relevantes
    #             priority_probabilities = [0.8 / len(priority_indices)] * len(priority_indices)
    #             normal_probabilities = [0.2 / len(self.indices)] * len(self.indices)
    #             combined_probabilities = [
    #                 priority_probabilities[i] if i in priority_indices else normal_probabilities[i]
    #                 for i in range(len(self.indices))
    #             ]
    #             indices = np.random.choice(self.indices, self.batch_size, replace=False, p=combined_probabilities)
    #             return indices

    #     # Selección normal si no aplicamos oversampling dinámico
    #     for b in range(self.batch_size):
    #         if self.current_position < len(self.indices):
    #             indices.append(self.indices[self.current_position])
    #             self.current_position += 1
    #         else:
    #             self.last_reached = True
    #             break

    #     if len(indices) > 0 and ((not self.last_reached) or self.return_incomplete):
    #         self.current_position += (self.number_of_threads_in_multithreaded - 1) * self.batch_size
    #         return indices
    #     else:
    #         self.reset()
    #         raise StopIteration

    
    
    def generate_train_batch(self):
        """
        Genera un lote de entrenamiento priorizando parches con la clase menos representada.
        """
        # logging.info("nnUNetDataLoader3DMinorityClass")
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        # Encuentra la clase menos representada (si oversampling dinámico está activado)
        least_represented_class = None
        if self.dynamic_oversampling:
            least_represented_class = self.get_least_represented_class()

        # print(f"Least represented class: {least_represented_class}")
        for j, i in enumerate(selected_keys):
            # Decide si sobresamplear foreground basado en la clase menos representada
            force_fg = False
            if self.dynamic_oversampling and least_represented_class is not None:
                force_fg = self.checking_least_represented_class(i)

            # Carga los datos y las etiquetas
            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # Calcula las coordenadas del parche
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'], least_represented_class)

            # Ajusta los límites del parche dentro del rango válido
            valid_bbox_lbs = np.clip(bbox_lbs, a_min=0, a_max=None)
            valid_bbox_ubs = np.minimum(shape, bbox_ubs)

            # Extrae el parche
            this_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            data = data[this_slice]

            this_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
            seg = seg[this_slice]

            # Aplica padding si es necesario
            padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
            padding = ((0, 0), *padding)
            data_all[j] = np.pad(data, padding, 'constant', constant_values=0)
            seg_all[j] = np.pad(seg, padding, 'constant', constant_values=-1)
        # Aplicar transformaciones
        if self.transforms is not None:
            with torch.no_grad():
                with threadpool_limits(limits=1, user_api=None):
                    data_all = torch.from_numpy(data_all).float()
                    seg_all = torch.from_numpy(seg_all).to(torch.int16)
                    images = []
                    segs = []
                    for b in range(self.batch_size):
                        tmp = self.transforms(**{'image': data_all[b], 'segmentation': seg_all[b]})
                        images.append(tmp['image'])
                        segs.append(tmp['segmentation'])
                    data_all = torch.stack(images)
                    if isinstance(segs[0], list):
                        seg_all = [torch.stack([s[i] for s in segs]) for i in range(len(segs[0]))]
                    else:
                        seg_all = torch.stack(segs)
                    del segs, images

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}

if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)

    # Selecciona el DataLoader
    use_dynamic = True  # Cambia a False para usar el estándar
    if use_dynamic:
        # print("Using nnUNetDataLoader3DMinorityClass:")
        dl = nnUNetDataLoader3DMinorityClass(ds, 5, (16, 16, 16), (16, 16, 16), dynamic_oversampling=True)
    else:
        # print("Using nnUNetDataLoader3D:")
        dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)

    batch = next(dl)
    # print("Batch generated successfully.")
