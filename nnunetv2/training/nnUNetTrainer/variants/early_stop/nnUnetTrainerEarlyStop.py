"""
This module contains the custom nnUNetTrainer classes with early stopping and oversampling support.
"""
from os.path import join
from typing import Optional, Dict, cast, Tuple
import torch
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.training.dataloading.data_loader_3d import MinorityClassOversampling_nnUNetDataLoader3D


class EarlyStopping:
    """
    EarlyStopping handler to stop training if no improvement is observed after a given number of epochs.
    """

    def __init__(self, patience: int, logger, min_delta: float = 0.0):
        """
        Initialize the EarlyStopping handler.

        Args:
            patience (int): Number of epochs to wait for improvement before stopping.
            logger: Logger object to log messages.
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
        Determine whether to stop training based on the current score.

        Args:
            current_score (float): The current score to compare against the best score.

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
        Save state of the early stopping.

        Returns:
            Dict[str, float]: Dictionary containing the state of early stopping.
        """
        return {"counter": self.counter, "best_score": cast(float, self.best_score)}

    def load_state_dict(self, state_dict: Dict[str, float]):
        """
        Load state of the early stopping.

        Args:
            state_dict (Dict[str, float]): Dictionary containing the state to load.
        """
        self.counter = state_dict["counter"]
        self.best_score = state_dict["best_score"]


class nnUNetTrainerBase(nnUNetTrainer):
    """
    Base Trainer with configurable hyperparameters and logging.
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
        Initialize the base trainer.

        Args:
            plans (dict): Training plans.
            configuration (str): Configuration name.
            fold (int): Fold number.
            dataset_json (dict): Dataset JSON.
            unpack_dataset (bool): Whether to unpack the dataset.
            device (torch.device): Device to use for training.
            initial_lr (float): Initial learning rate.
            weight_decay (float): Weight decay.
            oversample_foreground_percent (float): Percentage of foreground oversampling.
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
    """

    def __init__(self, *args, patience: int = 10, min_delta: float = 0.0, **kwargs):
        """
        Initialize the trainer with early stopping.

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
        Run the training process with early stopping.
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
    Trainer with Early Stopping and customizable oversampling percentages.
    """

    def __init__(self, *args, oversample_foreground_percent: float = 1.0, **kwargs):
        """
        Initialize the trainer with oversampling.

        Args:
            oversample_foreground_percent (float): Percentage of foreground oversampling.
        """
        super().__init__(*args, **kwargs)
        self.oversample_foreground_percent = oversample_foreground_percent
        self.logger.info(f"Final oversample percent: {self.oversample_foreground_percent}")


class nnUNetTrainerLowLROversampling(nnUNetTrainerOversampling):
    """
    Trainer with Early Stopping, low learning rate, and oversampling.
    """

    def __init__(self, *args, initial_lr: float = 5e-4, weight_decay: float = 1e-4, **kwargs):
        """
        Initialize the trainer with low learning rate and oversampling.

        Args:
            initial_lr (float): Initial learning rate.
            weight_decay (float): Weight decay.
        """
        super().__init__(*args, **kwargs)
        self.initial_lr = initial_lr
        self.weight_decay = weight_decay
        self.logger.info(f"Low LR: {self.initial_lr}, Higher Weight Decay: {self.weight_decay}")