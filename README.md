# Preserving Angles Improves Feature Distillation of Foundation Models

## Installation

Create a new Python 3.9 environment and install the packages in the requirements.txt file. 

## Feature Distillation

To fit a student network using the Cosine-similarity Preserving Compression (CosPress) approach run

	python3 main.py --config-name base.yaml +run=CosPress_imagenet_dinov2_vits.yaml 

For the Proteus approach, run

	python3 main.py --config-name base.yaml +run=proteus_imagenet_dinov2_vits.yaml 

This reproduces the approach outlined [here](https://github.com/BeSpontaneous/Proteus-pytorch) as closely as possible.

## Pretrained Weights

Pretrained weights can be downloaded from [here](https://figshare.com/s/193ace2befda8355fc79). Add them to the ./weights folder to run the scripts below.

## Evaluation

For the kNN and linear probe evaluations, run

	python3 main.py --config-name base.yaml +run=eval_knn.yaml 
	python3 main.py --config-name base.yaml +run=eval_dinov2_linear_probe.yaml 

For the semantic segmentation evaluation, run

	python3 main.py --config-name base.yaml +run=eval_semantic_segmentation.yaml 

For an example OOD detection evaluation, run

	python3 run/eval_ood_detection.py

To finetune distilled students using DeiT, run the following for the initial linear head training

	python3 main.py --config-name base.yaml +run=eval_deit_pre.yaml 

Then set XXX.ckpt in the config file to the previously trained checkpoint, and run

	python3 main.py --config-name base.yaml +run=eval_deit_post.yaml 







