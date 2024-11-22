"""This file prepares config fixtures for other tests."""

import pytest
from tests.helpers import helpers 
from pdb import set_trace as pb
from omegaconf import DictConfig, open_dict
from pathlib import Path

@pytest.fixture(scope="function")
def cfg_proteus(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=proteus", "model/dataset=torchvision_cifar100", "model/networks=proteus_vit_s_t"], 
        quicker=True)
    # cfg['pretrain'].model.trainer.max_epochs=1
    return cfg


@pytest.fixture(scope="function")
def cfg_deit_tiny_pretrained_cifar10(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=deit", "model/dataset=cifar10", "model/networks=deit_tiny_patch16_224_pretrained"],
        quicker=True)
    return cfg

@pytest.fixture(scope="function")
def cfg_supervised_segmentation(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=supervised_segmentation", "model/dataset=torchvision_voc2012", "model/networks=dinov2_vits_14_linear_classifier_segmentation"], 
        quicker=True)
    # cfg['pretrain'].model.trainer.max_epochs=1
    return cfg

@pytest.fixture(scope="function")
def cfg_dinov2_linear_finetune_224(tmp_path) -> DictConfig:
    cfg = {}
    cfg['pretrain'] = helpers.load_configuration_file(Path.joinpath(tmp_path,'pretrain'), 
        override = ["R=dinov2_linear_classifier", "model/dataset=imagenette", "model/networks=dinov2_vits_14_linear_classifier"], 
        quicker=True)
    # cfg['pretrain'].model.trainer.max_epochs=1
    return cfg