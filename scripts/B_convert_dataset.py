"""
This script converts the IMAGINEM dataset to the format required by nnU-Net v2, detailed in 'documentation/dataset_format.md'.
"""
import json
import os
import sys
import shutil

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.A_config import NNUNetConfig, DatasetType


if __name__ == "__main__":
    warning = "This script should only be ran once. " \
              "Are you sure you want to run it? [Y]/N \n"
    if input(warning).upper() != "Y":
        exit()

    # Create directories:
    try:
        os.mkdir(NNUNetConfig().dataset_raw_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(NNUNetConfig().train_images_dir)
    except FileExistsError:
        pass
    try:
        os.mkdir(NNUNetConfig().train_labels_dir)
    except FileExistsError:
        pass

    # Writing dataset JSON:
    dataset_json = {
        "channel_names": {
            "0": "Baseline",
            "1": "Follow-up"
        },
        "labels": {
            "background": 0,
            "Basal-lesions": 1,
            "New-lesions": 2
        },
        "numTraining": 81,
        "file_ending": ".nii.gz"
    }
    with open(NNUNetConfig().dataset_raw_dir / 'dataset.json', 'w') as f:
        json.dump(dataset_json, f)

    # Renaming of files:
    # Dividing files into masks and images:
    all_files_in_origin = [file_name for file_name in os.listdir(NNUNetConfig().ORIGINAL_DATA_PATH) if file_name.endswith(".nii.gz")]
    mask_files = [file_name for file_name in all_files_in_origin if "mask" in file_name]
    raw_files = [file_name for file_name in all_files_in_origin if "mask" not in file_name]

    # # Extracting ids:
    ids = [file_name.split(".")[0][:-5] for file_name in mask_files]

    # # Updating labels' names to nnU-Net format:
    label2newlabel = {
        label: label.split(".")[0][:-9] + '_01.nii.gz' for label in mask_files
    }
    for old_name, new_name in label2newlabel.items():
        os.rename(NNUNetConfig().ORIGINAL_DATA_PATH / old_name, NNUNetConfig().ORIGINAL_DATA_PATH / new_name)

    # Moving all files to the corresponding directory:
    for image in raw_files:
        shutil.copy(NNUNetConfig().ORIGINAL_DATA_PATH / image, NNUNetConfig().train_images_dir / image)
    for mask in label2newlabel.values():
        shutil.copy(NNUNetConfig().ORIGINAL_DATA_PATH / mask, NNUNetConfig().train_labels_dir / mask)
