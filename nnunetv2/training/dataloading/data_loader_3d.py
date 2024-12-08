import numpy as np
import torch
from threadpoolctl import threadpool_limits

from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class nnUNetDataLoader3D(nnUNetDataLoaderBase):
    def generate_train_batch(self):
        selected_keys = self.get_indices()
        # preallocate memory for data and seg
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

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

        return {'data': data_all, 'target': seg_all, 'keys': selected_keys}


def calculate_class_distribution(dataset):
    """
    Calcula la proporción de cada clase en el conjunto de datos.
    """
    all_labels = []
    for case in dataset.keys():
        label_file = dataset[case]['seg']
        label_data = torch.from_numpy(np.load(label_file)).flatten()
        all_labels.append(label_data)
    all_labels = torch.cat(all_labels)
    unique, counts = torch.unique(all_labels, return_counts=True)
    total_voxels = all_labels.numel()
    return {int(k): v.item() / total_voxels for k, v in zip(unique, counts)}


class nnUNetDataLoader3DMinorityClass(nnUNetDataLoaderBase):
    def __init__(self, *args, dynamic_oversampling=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_oversampling = dynamic_oversampling
        self.class_distribution = None
        if self.dynamic_oversampling:
            self.determine_class_distribution()

    def determine_class_distribution(self):
        """
        Calcula la distribución de clases en el conjunto de entrenamiento.
        """
        self.class_distribution = calculate_class_distribution(self.dataset)
        print(f"Class distribution: {self.class_distribution}")

    def get_least_represented_class(self):
        """
        Encuentra la clase menos representada.
        """
        if self.class_distribution is None:
            raise ValueError("Class distribution has not been calculated.")
        return min(self.class_distribution, key=self.class_distribution.get)

    def generate_train_batch(self):
        """
        Genera un lote de entrenamiento priorizando parches con la clase menos representada.
        """
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        # Encuentra la clase menos representada (si oversampling dinámico está activado)
        least_represented_class = None
        if self.dynamic_oversampling:
            least_represented_class = self.get_least_represented_class()

        print(f"Least represented class: {least_represented_class}")
        for j, i in enumerate(selected_keys):
            # Decide si sobresamplear foreground basado en la clase menos representada
            force_fg = False
            if self.dynamic_oversampling and least_represented_class is not None:
                case_class_locations = self.dataset[i]['class_locations']
                if least_represented_class in case_class_locations:
                    force_fg = True

            # Carga los datos y las etiquetas
            data, seg, properties = self._data.load_case(i)
            case_properties.append(properties)

            # Calcula las coordenadas del parche
            shape = data.shape[1:]
            dim = len(shape)
            bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'])

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
        print("Using nnUNetDataLoader3DMinorityClass:")
        dl = nnUNetDataLoader3DMinorityClass(ds, 5, (16, 16, 16), (16, 16, 16), dynamic_oversampling=True)
    else:
        print("Using nnUNetDataLoader3D:")
        dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)

    batch = next(dl)
    print("Batch generated successfully.")
