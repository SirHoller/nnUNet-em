"""
This module contains the custom nnUNetTrainer classes we have used.
"""

from os.path import join
from typing import Optional, Dict, Tuple

import numpy as np
import nibabel as nib
from threadpoolctl import threadpool_limits


import torch

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.sampling.nnUNetTrainer_probabilisticOversampling import (
    nnUNetTrainer_probabilisticOversampling,
)
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D, nnUNetDataLoader3DMinorityClass
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper


from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss, CustomWeightedLoss
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss




class EarlyStopping:
    """EarlyStopping handler to stop training if no improvement after a given number of events."""

    def __init__(self, patience: int, logger, min_delta: float = 0.0, cumulative_delta: bool = True):
        if patience < 1:
            raise ValueError("Argument patience should be a positive integer.")
        if min_delta < 0.0:
            raise ValueError("Argument min_delta should not be negative.")

        self.patience = patience
        self.min_delta = min_delta
        self.cumulative_delta = cumulative_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.logger = logger

    def stop_training(self, new_score: float) -> bool:
        if self.best_score is None:  # First epoch
            self.best_score = new_score
            return False

        if new_score <= self.best_score + self.min_delta:
            if not self.cumulative_delta and new_score > self.best_score:
                self.best_score = new_score
            self.counter += 1
            print(f"EarlyStopping: {self.counter} / {self.patience}")
            return self.counter >= self.patience
        else:  # New best score
            self.best_score = new_score
            self.counter = 0
            return False

    def state_dict(self) -> Dict[str, float]:
        return {"counter": self.counter, "best_score": self.best_score}

    def load_state_dict(self, state_dict: Dict[str, float]) -> None:
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]


class nnUNetTrainerEarlyStopping(nnUNetTrainer):
    """Variant of the nnU-Net Trainer with early stopping."""

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.early_stopping = EarlyStopping(
            patience=100, logger=self.logger, min_delta=0.01, cumulative_delta=False
        )
        self.print_to_log_file("Using early stopping with patience:", self.early_stopping.patience)

    def run_training(self):
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            
            self.on_train_epoch_start()
            train_outputs = []
            for batch_id in range(self.num_iterations_per_epoch):
                train_outputs.append(self.train_step(next(self.dataloader_train)))
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = []
                for batch_id in range(self.num_val_iterations_per_epoch):
                    val_outputs.append(self.validation_step(next(self.dataloader_val)))
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self.early_stopping.stop_training(new_score=self.logger.my_fantastic_logging['ema_fg_dice'][-1]):
                self.print_to_log_file("EarlyStopping: Stop training")
                break
            if self.early_stopping.counter % 10 == 0:
                self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        self.on_train_end()

class nnUNetTrainerEarlyStoppingWithOverSampling(nnUNetTrainer_probabilisticOversampling, nnUNetTrainerEarlyStopping):
    """Variant of the nnU-Net Trainer with early stopping and oversampling."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.5
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")
    
class nnUNetTrainerEarlyStoppingWithOverSampling_033(nnUNetTrainerEarlyStoppingWithOverSampling):
    """Variant of the nnU-Net Trainer with early stopping and oversampling."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 0.33
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")

class nnUNetTrainerEarlyStoppingWithOverSampling_100(nnUNetTrainerEarlyStoppingWithOverSampling):
    """Variant of the nnU-Net Trainer with early stopping and oversampling."""
    
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 1.0
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")

def calculate_class_distribution(labels):
    """
    Calcula la proporción de cada clase en un conjunto de etiquetas.
    """
    unique, counts = torch.unique(labels, return_counts=True)
    total_voxels = labels.numel()
    class_distribution = {int(k): v / total_voxels for k, v in zip(unique, counts)}
    return class_distribution


class nnUNetTrainerCustomOversamplingEarlyStopping(nnUNetTrainer_probabilisticOversampling, nnUNetTrainerEarlyStopping):
    """
    Entrenador de nnU-Net con early stopping y oversampling enfocado en la clase menos representada.
    """

    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 unpack_dataset: bool = True, device: torch.device = torch.device('cuda')):
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.oversample_foreground_percent = 1.0
        self.print_to_log_file(f"self.oversample_foreground_percent {self.oversample_foreground_percent}")
        
        
    # def _build_loss(self):
    #     if self.label_manager.has_regions:
    #         # Si hay regiones, puedes mantener la lógica actual con DC_and_BCE_loss
    #         loss = DC_and_BCE_loss(
    #             {},
    #             {
    #                 'batch_dice': self.configuration_manager.batch_dice,
    #                 'do_bg': True,
    #                 'smooth': 1e-5,
    #                 'ddp': self.is_ddp,
    #             },
    #             use_ignore_label=self.label_manager.ignore_label is not None,
    #             dice_class=MemoryEfficientSoftDiceLoss,
    #         )
    #     else:
    #         # Calcula los pesos de las clases a partir de las frecuencias
    #         class_counts = [100, 50, 5]  # Ejemplo de frecuencias de clases
    #         class_counts = [max(c, 1) for c in class_counts]  # Evitar divisiones por cero
    #         class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float32)
    #         class_weights = class_weights / class_weights.sum()  # Normaliza

    #         assert class_weights.dim() == 1, "class_weights debe ser un tensor unidimensional"
    #         class_weights = class_weights.cpu().numpy().tolist()  # Convertir a lista si es necesario

    #         # Crea una instancia de la nueva clase de pérdida
    #         loss = CustomWeightedLoss(
    #             class_weights=class_weights,
    #             weight_ce=1.0,  # Peso global para Cross Entropy Loss
    #             weight_dice=1.0,  # Peso global para Dice Loss
    #             ignore_label=self.label_manager.ignore_label
    #         )

    #     if self._do_i_compile() and hasattr(loss, 'dice_loss'):
    #         # Aplica compilación a Dice Loss si es compatible
    #         loss.dice_loss = torch.compile(loss.dice_loss)

    #     if self.enable_deep_supervision:
    #         # Configura Deep Supervision
    #         deep_supervision_scales = self._get_deep_supervision_scales()
    #         weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])

    #         if self.is_ddp and not self._do_i_compile():
    #             weights[-1] = 1e-6
    #         else:
    #             weights[-1] = 0

    #         weights = weights / weights.sum()
    #         loss = DeepSupervisionWrapper(loss, weights)

    #     return loss


    def get_dataloaders(self):
        self.print_to_log_file("nnUNetTrainerCustomOversamplingEarlyStopping, get_dataloaders")
        patch_size = self.configuration_manager.patch_size
        dim = len(patch_size)

        # needed for deep supervision: how much do we need to downscale the segmentation targets for the different
        # outputs?

        deep_supervision_scales = self._get_deep_supervision_scales()

        (
            rotation_for_DA,
            do_dummy_2d_data_aug,
            initial_patch_size,
            mirror_axes,
        ) = self.configure_rotation_dummyDA_mirroring_and_inital_patch_size()

        # training pipeline
        tr_transforms = self.get_training_transforms(
            patch_size, rotation_for_DA, deep_supervision_scales, mirror_axes, do_dummy_2d_data_aug,
            use_mask_for_norm=self.configuration_manager.use_mask_for_norm,
            is_cascaded=self.is_cascaded, foreground_labels=self.label_manager.foreground_labels,
            regions=self.label_manager.foreground_regions if self.label_manager.has_regions else None,
            ignore_label=self.label_manager.ignore_label)

        # validation pipeline
        val_transforms = self.get_validation_transforms(deep_supervision_scales,
                                                        is_cascaded=self.is_cascaded,
                                                        foreground_labels=self.label_manager.foreground_labels,
                                                        regions=self.label_manager.foreground_regions if
                                                        self.label_manager.has_regions else None,
                                                        ignore_label=self.label_manager.ignore_label)

        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader2D(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)
        else:
            dl_tr = nnUNetDataLoader3DMinorityClass(dataset_tr, self.batch_size,
                                       initial_patch_size,
                                       self.configuration_manager.patch_size,
                                       self.label_manager,
                                       oversample_foreground_percent=self.oversample_foreground_percent,
                                       sampling_probabilities=None, pad_sides=None, transforms=tr_transforms)
            dl_val = nnUNetDataLoader3DMinorityClass(dataset_val, self.batch_size,
                                        self.configuration_manager.patch_size,
                                        self.configuration_manager.patch_size,
                                        self.label_manager,
                                        oversample_foreground_percent=self.oversample_foreground_percent,
                                        sampling_probabilities=None, pad_sides=None, transforms=val_transforms)

        allowed_num_processes = get_allowed_n_proc_DA()
        if allowed_num_processes == 0:
            mt_gen_train = SingleThreadedAugmenter(dl_tr, None)
            mt_gen_val = SingleThreadedAugmenter(dl_val, None)
        else:
            mt_gen_train = NonDetMultiThreadedAugmenter(data_loader=dl_tr, transform=None,
                                                        num_processes=allowed_num_processes,
                                                        num_cached=max(6, allowed_num_processes // 2), seeds=None,
                                                        pin_memory=self.device.type == 'cuda', wait_time=0.002)
            mt_gen_val = NonDetMultiThreadedAugmenter(data_loader=dl_val,
                                                      transform=None, num_processes=max(1, allowed_num_processes // 2),
                                                      num_cached=max(3, allowed_num_processes // 4), seeds=None,
                                                      pin_memory=self.device.type == 'cuda',
                                                      wait_time=0.002)
        # # let's get this party started
        _ = next(mt_gen_train)
        _ = next(mt_gen_val)
        return mt_gen_train, mt_gen_val

    # def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
    #     print(f"nnUNetTrainerCustomOversamplingEarlyStopping, get_plain_dataloaders")
    #     dataset_tr, dataset_val = self.get_tr_and_val_datasets()

    #     if dim == 2:
    #         dl_tr = nnUNetDataLoader2D(dataset_tr,
    #                                    self.batch_size,
    #                                    initial_patch_size,
    #                                    self.configuration_manager.patch_size,
    #                                    self.label_manager,
    #                                    oversample_foreground_percent=self.oversample_foreground_percent,
    #                                    sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
    #         dl_val = nnUNetDataLoader2D(dataset_val,
    #                                     self.batch_size,
    #                                     self.configuration_manager.patch_size,
    #                                     self.configuration_manager.patch_size,
    #                                     self.label_manager,
    #                                     oversample_foreground_percent=self.oversample_foreground_percent,
    #                                     sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
    #     else:
    #         dl_tr = nnUNetDataLoader3DMinorityClass(dataset_tr,
    #                                    self.batch_size,
    #                                    initial_patch_size,
    #                                    self.configuration_manager.patch_size,
    #                                    self.label_manager,
    #                                    oversample_foreground_percent=self.oversample_foreground_percent,
    #                                    sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
    #         dl_val = nnUNetDataLoader3DMinorityClass(dataset_val,
    #                                     self.batch_size,
    #                                     self.configuration_manager.patch_size,
    #                                     self.configuration_manager.patch_size,
    #                                     self.label_manager,
    #                                     oversample_foreground_percent=self.oversample_foreground_percent,
    #                                     sampling_probabilities=None, pad_sides=None, probabilistic_oversampling=True)
    #     return dl_tr, dl_val
