@echo off
REM Paths:
set nnUNet_raw=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_raw_data
set nnUNet_preprocessed=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_preprocessed_data
set nnUNet_results=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_results

REM Preprocess dataset
nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

REM Train folds (we could also use "all" to train all splits):
set dataset=100
set conf=3d_fullres
set trainer=nnUNetTrainerCustomOversamplingEarlyStopping

@REM nnUNetv2_train %dataset% %conf% 0 -device cuda -tr %trainer% --npz --val_best=1
@REM nnUNetv2_train %dataset% %conf% 1 -device cuda -tr %trainer% --npz --val_best=1
@REM nnUNetv2_train %dataset% %conf% 2 -device cuda -tr %trainer% --npz --val_best=1
@REM nnUNetv2_train %dataset% %conf% 3 -device cuda -tr %trainer% --npz --val_best=1
nnUNetv2_train %dataset% %conf% 4 -device cuda -tr %trainer% --npz

REM Find best configuration:
nnUNetv2_find_best_configuration %dataset% -c %conf% -tr %trainer%

REM Predict
set INPUT_FOLDER=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_raw_data\Dataset100_NewLesions\imagesTs
set OUTPUT_FOLDER=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_test_results\Dataset100_NewLesions\nnUNetTrainerCustomOversamplingEarlyStopping__nnUNetPlans__3d_fullres
nnUNetv2_predict -d %dataset% -i %INPUT_FOLDER% -o %OUTPUT_FOLDER% -f 0 1 2 3 4 -tr %trainer% -c %conf% -p nnUNetPlans