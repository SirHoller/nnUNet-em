dataset=100
conf=3d_fullres
trainer=nnUNetTrainerCustomOversamplingEarlyStopping

INPUT_FOLDER=./input
OUTPUT_FOLDER=./output
# Predict
nnUNetv2_predict -d $dataset -i $INPUT_FOLDER -o $OUTPUT_FOLDER -f  0 1 2 3 4 -tr $trainer -c $conf -p nnUNetPlans

rm plans.json