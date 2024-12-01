"""
Custom nnUNetTrainer classes with Early Stopping, Oversampling, and flexible configurations.
"""
from typing import Tuple, Dict, Optional, cast
from os.path import join
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import MinorityClassOversampling_nnUNetDataLoader3D


class EarlyStopping:
    """
    EarlyStopping handler to stop training if no improvement is observed after a given number of epochs.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change to qualify as an improvement.
        best_score (Optional[float]): Best score observed so far.
        counter (int): Number of epochs since the last improvement.
        logger: Logger for logging messages.
    """

    def __init__(self, patience: int, logger, min_delta: float = 0.0):
        """
        Initializes the EarlyStopping handler.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            logger: Logger for logging messages.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        if patience < 1:
            raise ValueError("patience must be a positive integer.")
        if min_delta < 0.0:
            raise ValueError("min_delta cannot be negative.")
        self.patience = patience
        self.min_delta = min_delta
        self.best_score: Optional[float] = None
        self.counter = 0
        self.logger = logger

    def stop_training(self, current_score: float) -> bool:
        """
        Determines whether training should be stopped based on the current score.

        Args:
            current_score (float): The current score to evaluate.

        Returns:
            bool: True if training should be stopped, False otherwise.
        """
        if self.best_score is None:
            self.best_score = current_score
            return False

        if current_score <= self.best_score + self.min_delta:
            self.counter += 1
            self.logger.info(f"No improvement. Patience {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = current_score
            self.counter = 0
        return False

    def state_dict(self) -> Dict[str, float]:
        """
        Returns the state of the EarlyStopping handler.

        Returns:
            Dict[str, float]: The state dictionary containing the counter and best score.
        """
        return {"counter": self.counter, "best_score": cast(float, self.best_score)}

    def load_state_dict(self, state_dict: Dict[str, float]):
        """
        Loads the state of the EarlyStopping handler from a state dictionary.

        Args:
            state_dict (Dict[str, float]): The state dictionary containing the counter and best score.
        """
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]


class nnUNetTrainerBase(nnUNetTrainer):
    """
    Base Trainer with configurable hyperparameters and logging.

    Attributes:
        initial_lr (float): Initial learning rate.
        weight_decay (float): Weight decay for the optimizer.
        oversample_foreground_percent (float): Percentage of oversampling for foreground pixels.
    """

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        unpack_dataset: bool = True,
        device: torch.device = torch.device("cuda"),
        initial_lr: float = 1e-3,
        weight_decay: float = 0.0,
        oversample_foreground_percent: float = 0.5,
    ):
        """
        Initializes the nnUNetTrainerBase.

        Args:
            plans (dict): Plans for the training.
            configuration (str): Configuration name.
            fold (int): Fold number.
            dataset_json (dict): Dataset JSON.
            unpack_dataset (bool): Whether to unpack the dataset.
            device (torch.device): Device to use for training.
            initial_lr (float): Initial learning rate.
            weight_decay (float): Weight decay for the optimizer.
            oversample_foreground_percent (float): Percentage of oversampling for foreground pixels.
        """
        super().__init__(plans, configuration, fold, dataset_json, unpack_dataset, device)
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.oversample_foreground_percent = oversample_foreground_percent
        self.logger.info(f"Initial LR: {self.initial_lr}, Weight Decay: {self.weight_decay}")
        self.logger.info(f"Oversample Foreground Percent: {self.oversample_foreground_percent}")


class nnUNetTrainerWithEarlyStopping(nnUNetTrainerBase):
    """
    Trainer with Early Stopping support.

    Attributes:
        early_stopping (EarlyStopping): EarlyStopping handler.
    """

    def __init__(self, *args, patience: int = 10, min_delta: float = 0.0, **kwargs):
        """
        Initializes the nnUNetTrainerWithEarlyStopping.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            min_delta (float): Minimum change to qualify as an improvement.
        """
        super().__init__(*args, **kwargs)
        self.early_stopping = EarlyStopping(
            patience=patience, logger=self.logger, min_delta=min_delta
        )
        self.logger.info(f"Using Early Stopping with patience: {patience}, min_delta: {min_delta}")

    def run_training(self):
        """
        Runs the training process with early stopping.
        """
        self.on_train_start()
        for epoch in range(self.current_epoch, self.num_epochs):
            self.on_epoch_start()

            self.on_train_epoch_start()
            train_outputs = [
                self.train_step(next(self.dataloader_train))
                for _ in range(self.num_iterations_per_epoch)
            ]
            self.on_train_epoch_end(train_outputs)

            with torch.no_grad():
                self.on_validation_epoch_start()
                val_outputs = [
                    self.validation_step(next(self.dataloader_val))
                    for _ in range(self.num_val_iterations_per_epoch)
                ]
                self.on_validation_epoch_end(val_outputs)

            self.on_epoch_end()
            current_score = self.logger.my_fantastic_logging["ema_fg_dice"][-1]
            if self.early_stopping.stop_training(current_score):
                self.logger.info("EarlyStopping: Stopping training")
                break
            elif self.early_stopping.counter % 10 == 0:
                self.save_checkpoint(join(self.output_folder, "checkpoint_latest.pth"))
        self.on_train_end()


class nnUNetTrainerOversampling(nnUNetTrainerWithEarlyStopping):
    """
    Trainer with configurable oversampling for foreground pixels.

    Attributes:
        oversample_foreground_percent (float): Percentage of oversampling for foreground pixels.
    """

    def __init__(self, *args, oversample_foreground_percent: float = 0.5, **kwargs):
        """
        Initializes the nnUNetTrainerOversampling.

        Args:
            oversample_foreground_percent (float): Percentage of oversampling for foreground pixels.
        """
        super().__init__(*args, **kwargs)
        self.oversample_foreground_percent = oversample_foreground_percent
        self.logger.info(f"Oversample foreground percent set to {self.oversample_foreground_percent * 100}%.")

    def get_plain_dataloaders(self, initial_patch_size: Tuple[int, ...], dim: int):
        """
        Gets the plain dataloaders for training and validation.

        Args:
            initial_patch_size (Tuple[int, ...]): Initial patch size.
            dim (int): Dimension of the data (2D or 3D).

        Returns:
            Tuple: Training and validation dataloaders.
        """
        dataset_tr, dataset_val = self.get_tr_and_val_datasets()

        if dim == 2:
            dl_tr = nnUNetDataLoader2D(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent
            )
            dl_val = nnUNetDataLoader2D(
                dataset_val, self.batch_size, self.configuration_manager.patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent
            )
        else:
            dl_tr = MinorityClassOversampling_nnUNetDataLoader3D(
                dataset_tr, self.batch_size, initial_patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent
            )
            dl_val = MinorityClassOversampling_nnUNetDataLoader3D(
                dataset_val, self.batch_size, self.configuration_manager.patch_size, self.configuration_manager.patch_size,
                self.label_manager, oversample_foreground_percent=self.oversample_foreground_percent
            )

        return dl_tr, dl_val


class nnUNetTrainerExtremeOversampling(nnUNetTrainerOversampling):
    """
    Trainer with extreme oversampling (100% of foreground patches).
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the nnUNetTrainerExtremeOversampling with 100% foreground oversampling.
        """
        super().__init__(*args, oversample_foreground_percent=1.0, **kwargs)
        self.logger.info("Using extreme oversampling (100% foreground) in dataloaders.")