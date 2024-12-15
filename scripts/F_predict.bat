@echo off
REM Paths:
set nnUNet_raw=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_raw_data
set nnUNet_preprocessed=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_preprocessed_data
set nnUNet_results=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_results

REM Preprocess dataset
@REM nnUNetv2_plan_and_preprocess -d 100 --verify_dataset_integrity

REM Train folds (we could also use "all" to train all splits):
set dataset=100
set conf=3d_fullres
set trainer=nnUNetTrainerCustomOversamplingEarlyStopping

REM Predict
set INPUT_FOLDER=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_raw_data\Dataset101_NewLesionsChallenge\imagesTs
set OUTPUT_FOLDER=C:\Users\mrtwe\TFM\nnUNet-em\nnUNet_test_results\Dataset101_NewLesionsChallenge\nnUNetTrainerCustomOversamplingEarlyStopping__nnUNetPlans__3d_fullres
nnUNetv2_predict -d %dataset% -i %INPUT_FOLDER% -o %OUTPUT_FOLDER% -f 0 1 2 3 4 -tr %trainer% -c %conf% -p nnUNetPlans