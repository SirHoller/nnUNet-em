"""
This module contains the custom nnUNetTrainer classes we have used.
"""

from os.path import join
from typing import Optional, Dict, Tuple

import torch
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import MinorityClassOversampling_nnUNetDataLoader3D
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.nnUNetTrainer.variants.sampling.nnUNetTrainer_probabilisticOversampling import (
    nnUNetTrainer_probabilisticOversampling,
)


class EarlyStopping:
    """EarlyStopping handler to stop training if no improvement after a given number of events."""

    def __init__(self, patience: int, logger, min_delta: float = 0.0, cumulative_delta: bool = False):
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
            patience=100, logger=self.logger, min_delta=0, cumulative_delta=False
        )
        self.print_to_log_file("Using early stopping with patience:", self.early_stopping.patience)

    def run_training(self):
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()
            train_outputs = [self.train_step(next(self.dataloader_train)) for _ in range(self.num_iterations_per_epoch)]
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [self.validation_step(next(self.dataloader_val)) for _ in range(self.num_val_iterations_per_epoch)]
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            if self.early_stopping.stop_training(new_score=self.logger.my_fantastic_logging['ema_fg_dice'][-1]):
                self.print_to_log_file("EarlyStopping: Stop training")
                break
            elif self.early_stopping.counter % 10 == 0:
                self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        self.on_train_end()


class OversamplingTrainerMixin:
    """Mixin for trainers with configurable oversampling."""

    def __init__(self, oversample_foreground_percent: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.oversample_foreground_percent = oversample_foreground_percent
        self.print_to_log_file("Oversample percent:", self.oversample_foreground_percent)


class nnUNetTrainerHalfOversamplingEarlyStopping(OversamplingTrainerMixin, nnUNetTrainerEarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(oversample_foreground_percent=0.5, *args, **kwargs)


class nnUNetTrainerFullOversamplingEarlyStopping(OversamplingTrainerMixin, nnUNetTrainerEarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(oversample_foreground_percent=1, *args, **kwargs)


class nnUNetTrainerExtremeOversamplingEarlyStopping(nnUNetTrainerFullOversamplingEarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.print_to_log_file("Using minority class oversampling.")

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()
        dataloader_cls = nnUNetDataLoader2D if dim == 2 else MinorityClassOversampling_nnUNetDataLoader3D
        dl_tr = dataloader_cls(dataset_tr, self.batch_size, initial_patch_size,
                               self.configuration_manager.patch_size, self.label_manager,
                               oversample_foreground_percent=self.oversample_foreground_percent)
        dl_val = dataloader_cls(dataset_val, self.batch_size, self.configuration_manager.patch_size,
                                self.configuration_manager.patch_size, self.label_manager,
                                oversample_foreground_percent=self.oversample_foreground_percent)
        return dl_tr, dl_val


class LearningRateDecayMixin:
    """Mixin for configuring learning rate and weight decay."""

    def __init__(self, initial_lr: float, weight_decay: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.print_to_log_file("Initial lr:", self.initial_lr)
        self.print_to_log_file("Weight decay:", self.weight_decay)


class nnUNetTrainerExtremeOversamplingEarlyStoppingLowLR(LearningRateDecayMixin, nnUNetTrainerExtremeOversamplingEarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(initial_lr=1e-3, *args, **kwargs)


class nnUNetTrainerExtremeOversamplingEarlyStoppingVeryLowLR(LearningRateDecayMixin, nnUNetTrainerExtremeOversamplingEarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(initial_lr=5e-4, *args, **kwargs)


class nnUNetTrainerExtremeOversamplingEarlyStoppingVeryLowLRVeryHighDecay(LearningRateDecayMixin, nnUNetTrainerExtremeOversamplingEarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(initial_lr=5e-4, weight_decay=5e-4, *args, **kwargs)


class nnUNetTrainerExtremeOversamplingEarlyStoppingLowLRHigherDecay(LearningRateDecayMixin, nnUNetTrainerExtremeOversamplingEarlyStopping):
    def __init__(self, *args, **kwargs):
        super().__init__(initial_lr=1e-3, weight_decay=1e-4, *args, **kwargs)
