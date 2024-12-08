"""
Module for nnU-Net configuration variables and paths:
    1. Defines paths for nnU-Net and sets them as environment variables.
    2. Centralizes dataset-specific configurations and fixed variables.
    3. Provides structured access to train, test.
"""

import os
from enum import Enum
from pathlib import Path
import pandas as pd

# Pandas Configuration
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 900)

class NNUNetConfig:
    """
    Configuration class for managing nnU-Net paths and settings.
    """
    ROOT = Path(__file__).resolve().parents[1]
    
    # General Paths
    ORIGINAL_DATA_PATH = ROOT / "NEW_LESIONS_IMAGINEM"
    RAW_DATA_PATH = ROOT / "nnUNet_raw_data"
    PREPROCESSED_DATA_PATH = ROOT / "nnUNet_preprocessed_data"
    RESULTS_PATH = ROOT / "nnUNet_results"
    TEST_RESULTS_PATH = ROOT / "nnUNet_test_results"

    # Dataset-specific Settings
    DATASET_NAME = "Dataset100_NewLesions"
    TERMINATION = ".nii.gz"
    CONFIGURATION = "3d_fullres"
    PLAN = "nnUNetPlans"

    @property
    def dataset_raw_dir(self) -> str:
        return self.RAW_DATA_PATH / self.DATASET_NAME

    @property
    def dataset_preprocessed_dir(self) -> str:
        return self.PREPROCESSED_DATA_PATH / self.DATASET_NAME

    # Train/Test Split Paths
    @property
    def train_images_dir(self) -> str:
        return self.dataset_raw_dir / "imagesTr"

    @property
    def train_labels_dir(self) -> str:
        return self.dataset_raw_dir / "labelsTr"

    @property
    def test_images_dir(self) -> str:
        return self.dataset_raw_dir / "imagesTs"

    @property
    def test_labels_dir(self) -> str:
        return self.dataset_raw_dir / "labelsTs"

    def export_paths_to_env(self):
        """
        Exports nnU-Net paths as environment variables for seamless integration.
        """
        os.environ['nnUNet_raw'] = str(self.RAW_DATA_PATH)
        os.environ['nnUNet_preprocessed'] = str(self.PREPROCESSED_DATA_PATH)
        os.environ['nnUNet_results'] = str(self.RESULTS_PATH)
        print(f"Environment variables set: nnUNet_raw, nnUNet_preprocessed, nnUNet_results")
        print(str(self.RAW_DATA_PATH))
        print(str(self.PREPROCESSED_DATA_PATH))
        print(str(self.RESULTS_PATH))


class DatasetType(str, Enum):
    """
    Enum for dataset types.
    """
    TRAIN_SPLIT = "train_split"
    TEST_SPLIT = "test_split"

