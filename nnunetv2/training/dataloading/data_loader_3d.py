import numpy as np
from nnunetv2.training.dataloading.base_data_loader import nnUNetDataLoaderBase
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset


class BaseDataLoader3D(nnUNetDataLoaderBase):
    def process_case(self, data, seg, properties, force_fg, overwrite_class=None):
        shape = data.shape[1:]
        dim = len(shape)
        bbox_lbs, bbox_ubs = self.get_bbox(
            data_shape=shape,
            force_fg=force_fg,
            class_locations=properties['class_locations'],
            overwrite_class=overwrite_class
        )

        # Crop to valid bounding box
        valid_bbox_lbs = [max(0, bbox_lbs[i]) for i in range(dim)]
        valid_bbox_ubs = [min(shape[i], bbox_ubs[i]) for i in range(dim)]

        # Slicing for data and segmentation
        data_slice = tuple([slice(0, data.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        seg_slice = tuple([slice(0, seg.shape[0])] + [slice(i, j) for i, j in zip(valid_bbox_lbs, valid_bbox_ubs)])
        data = data[data_slice]
        seg = seg[seg_slice]

        # Padding
        padding = [(-min(0, bbox_lbs[i]), max(bbox_ubs[i] - shape[i], 0)) for i in range(dim)]
        data_padded = np.pad(data, ((0, 0), *padding), 'constant', constant_values=0)
        seg_padded = np.pad(seg, ((0, 0), *padding), 'constant', constant_values=-1)

        return data_padded, seg_padded

    def generate_train_batch(self, overwrite_class=None):
        selected_keys = self.get_indices()
        data_all = np.zeros(self.data_shape, dtype=np.float32)
        seg_all = np.zeros(self.seg_shape, dtype=np.int16)
        case_properties = []

        for j, i in enumerate(selected_keys):
            force_fg = self.get_do_oversample(j)
            data, seg, properties = self._data.load_case(i)
            data_padded, seg_padded = self.process_case(data, seg, properties, force_fg, overwrite_class)
            data_all[j] = data_padded
            seg_all[j] = seg_padded

        return {'data': data_all, 'seg': seg_all, 'properties': case_properties, 'keys': selected_keys}


class nnUNetDataLoader3D(BaseDataLoader3D):
    pass


class MinorityClassOversampling_nnUNetDataLoader3D(BaseDataLoader3D):
    def generate_train_batch(self):
        return super().generate_train_batch(overwrite_class=2)


if __name__ == '__main__':
    folder = '/media/fabian/data/nnUNet_preprocessed/Dataset002_Heart/3d_fullres'
    ds = nnUNetDataset(folder, 0)
    dl = nnUNetDataLoader3D(ds, 5, (16, 16, 16), (16, 16, 16), 0.33, None, None)
    a = next(dl)
