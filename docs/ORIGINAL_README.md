# Contrastive Distillation and Cross-Modal Mamba Fusion for Multimodal Brain Disease Diagnosis

This repository contains source code for "Contrastive Distillation and Cross-Modal Mamba Fusion for Multimodal Brain Disease Diagnosis". 

## Dependencies
- python=3.8
- torchvision=0.15.0+cu118
- torch=2.0.0+cu118
- torchaudio=2.0.0+cu118
- wandb=0.17.2
- scikit-learn=1.3.2
- pandas=2.0.3
- mamba-ssm 2.2.2

## Usage
Run the following command to train the model.

`python -m source --multirun datasz=100p model=CDCo-MambaNet dataset=UCLA repeat_time=5 preprocess=mixup`

## Data Availability

The repository uses public neuroimaging datasets. Raw data must be obtained from the official sources and used in accordance with their licenses/terms:

ADNI: https://adni.loni.usc.edu/

PPMI: https://www.ppmi-info.org/

UCLA (LA5c): https://legacy.openfmri.org/dataset/ds000030/

ABIDE: http://preprocessed-connectomes-project.org/abide/

TaoWu & Neurocon (PD): http://fcon_1000.projects.nitrc.org/indi/retro/parkinsons.html

## Preprocessing Overview

Two preprocessing routes are assumed:

rs-fMRI: DPABI/DPARSFA-based preprocessing followed by ROI time-series (ROISignals) extraction.

sMRI (T1w): SimpleITK-based preprocessing including bias correction, (optional) skull stripping, MNI normalization, and reshaping to 96×96×96.

## Pretrained Weights

Pretrained weights are provided here:

Weights: https：https://pan.baidu.com/s/1PeayeTF_AYCfsm4pZKzW_Q Key:2772

