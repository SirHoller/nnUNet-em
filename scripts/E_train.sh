# Paths:
export nnUNet_raw="/Users/castiler/TFM/nnUNet-em/nnUNet-em/nnUNet_raw_data"
export nnUNet_preprocessed="/Users/castiler/TFM/nnUNet-em/nnUNet-em/nnUNet_preprocessed_data"
export nnUNet_results="/Users/castiler/TFM/nnUNet-em/nnUNet-em/nnUNet_results"

# Preprocess dataset
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

# Train folds (we could also use "all" to train all splits):
dataset=100
conf=3d_fullres
trainer=nnUNetTrainerCustomOversamplingEarlyStopping
nnUNetv2_train $dataset $conf 0 -device cuda -tr $trainer --npz
nnUNetv2_train $dataset $conf 1 -device cuda -tr $trainer --npz
nnUNetv2_train $dataset $conf 2 -device cuda -tr $trainer --npz
nnUNetv2_train $dataset $conf 3 -device cuda -tr $trainer --npz
nnUNetv2_train $dataset $conf 4 -device cuda -tr $trainer --npz

# Find best configuration:
nnUNetv2_find_best_configuration $dataset -c $conf -tr $trainer

# Predict
INPUT_FOLDER=/Users/castiler/TFM/nnUNet-em/nnUNet_raw_data/Dataset100_NewLesions/imagesTs
OUTPUT_FOLDER=/Users/castiler/TFM/nnUNet-em/nnUNet_test_results/Dataset100_NewLesions/nnUNetTrainerCustomOversamplingEarlyStopping__nnUNetPlans__3d_fullres
nnUNetv2_predict -d $dataset -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 4 -tr $trainer -c $conf -p nnUNetPlans