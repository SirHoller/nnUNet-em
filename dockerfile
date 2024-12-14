FROM python:3.10-slim
RUN apt-get update -y && apt-get upgrade -y
RUN apt-get install --no-install-recommends -y build-essential

WORKDIR /nnUNet-em

RUN pip3 install --upgrade pip

COPY nnunetv2 nnunetv2
COPY setup.py setup.py
COPY requirements.txt requirements.txt

ARG model_folder=/nnUNet_results/Dataset100_NewLesions/nnUNetTrainerCustomOversamplingEarlyStopping__nnUNetPlans__3d_fullres
RUN mkdir -p $model_folder

COPY $model_folder/*.json $model_folder/
COPY $model_folder/fold_0/checkpoint_best.pth $model_folder/fold_0/checkpoint_final.pth
COPY $model_folder/fold_1/checkpoint_best.pth $model_folder/fold_1/checkpoint_final.pth
COPY $model_folder/fold_2/checkpoint_best.pth $model_folder/fold_2/checkpoint_final.pth
COPY $model_folder/fold_3/checkpoint_best.pth $model_folder/fold_3/checkpoint_final.pth
COPY $model_folder/fold_4/checkpoint_best.pth $model_folder/fold_4/checkpoint_final.pth

RUN mkdir -p /output
RUN mkdir -p /input

RUN ls -ls

RUN pip3 install -r requirements.txt

RUN pip3 install -e .

ENV nnUNet_raw='/nnUNet_raw_data'
ENV nnUNet_preprocessed='/nnUNet_preprocessed'
ENV nnUNet_results='/nnUNet_results'

COPY predict.sh /predict.sh
