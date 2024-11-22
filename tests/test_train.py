import os

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import open_dict

from main import train
from tests.helpers.run_if import RunIf
from tests.helpers import helpers 
from tests.helpers import seetests

from pdb import set_trace as pb
from lightning.pytorch.callbacks import ModelCheckpoint
import numpy as np

from torch import tensor
import torch
nan = float('nan')

# pytest -s tests/test_train.py::test_proteus_fast_dev_run_gpu
@RunIf(min_gpus=1)
def test_proteus_fast_dev_run_gpu(cfg_proteus):
    prev_results_train = {'lr-AdamW/pg1': tensor(0.0001), 'lr-AdamW/pg1-momentum': tensor(0.9000), 'lr-AdamW/pg2': tensor(0.0001), 'lr-AdamW/pg2-momentum': tensor(0.9000), 'train/loss': tensor(17.7274), 'train/loss_step': tensor(17.7274), 'train/loss_epoch': tensor(17.7274), 'train/patch_loss': tensor(5.8737), 'train/fea_loss': tensor(5.7330), 'train/token_loss': tensor(6.1207), 'val/loss': tensor(16.5831), 'val/patch_loss': tensor(5.2315), 'val/fea_loss': tensor(5.5846), 'val/token_loss': tensor(5.7670), 'test/loss': tensor(15.6698), 'test/patch_loss': tensor(4.9857), 'test/fea_loss': tensor(5.3131), 'test/token_loss': tensor(5.3709)}
    cfg = cfg_proteus
    with open_dict(cfg['pretrain']):
        cfg['pretrain']['model']['trainer']['deterministic'] = False
        cfg['pretrain']['model']['trainer']['precision'] = '16-mixed'
        cfg['pretrain']['model']['trainer']['num_sanity_val_steps'] = 0
        # cfg['pretrain']['model']['dataloaders']['batch_size'] = 32
    seetests.pretrain_eval_finetune_test(cfg, prev_results_train, accelerator='gpu', check_keys_train=False) # can't do gpu as non-deterministic

# pytest -s tests/test_train.py::test_deit_tiny_pretrained_cifar10_fast_dev_run_gpu
def test_deit_tiny_pretrained_cifar10_fast_dev_run_gpu(cfg_deit_tiny_pretrained_cifar10):
    prev_results_train = {'lr-AdamW/pg1': tensor(9.3750e-06, dtype=torch.float64), 'lr-AdamW/pg1-momentum': tensor(0.9000), 'lr-AdamW/pg2': tensor(9.3750e-06, dtype=torch.float64), 'lr-AdamW/pg2-momentum': tensor(0.9000), 'lr-AdamW/pg3': tensor(9.3750e-06, dtype=torch.float64), 'lr-AdamW/pg3-momentum': tensor(0.9000), 'lr-AdamW/pg4': tensor(9.3750e-06, dtype=torch.float64), 'lr-AdamW/pg4-momentum': tensor(0.9000), 'train/loss': tensor(2.3249), 'train/loss_step': tensor(2.3249), 'train/loss_epoch': tensor(2.3249), 'train/accuracy': tensor(0.), 'train/mean_class_accuracy': tensor(0.), 'val/loss': tensor(2.4316), 'val/accuracy': tensor(0.), 'val/mean_class_accuracy': tensor(0.), 'val/aucroc': tensor(0.0500), 'val/average_precision_0': tensor(0.2500), 'val/average_precision_1': tensor(nan), 'val/average_precision_2': tensor(nan), 'val/average_precision_3': tensor(0.2500), 'val/average_precision_4': tensor(nan), 'val/average_precision_5': tensor(nan), 'val/average_precision_6': tensor(nan), 'val/average_precision_7': tensor(nan), 'val/average_precision_8': tensor(0.5833), 'val/average_precision_9': tensor(nan), 'test/loss': tensor(2.4316), 'test/accuracy': tensor(0.), 'test/mean_class_accuracy': tensor(0.), 'test/aucroc': tensor(0.0500), 'test/average_precision_0': tensor(0.2500), 'test/average_precision_1': tensor(nan), 'test/average_precision_2': tensor(nan), 'test/average_precision_3': tensor(0.2500), 'test/average_precision_4': tensor(nan), 'test/average_precision_5': tensor(nan), 'test/average_precision_6': tensor(nan), 'test/average_precision_7': tensor(nan), 'test/average_precision_8': tensor(0.5833), 'test/average_precision_9': tensor(nan)}
    seetests.pretrain_eval_finetune_test(cfg_deit_tiny_pretrained_cifar10, prev_results_train, accelerator='gpu', check_keys_train=False)

# pytest -s tests/test_train.py::test_supervised_segmentation_fast_dev_run_gpu
@RunIf(min_gpus=1)
def test_supervised_segmentation_fast_dev_run_gpu(cfg_supervised_segmentation):
    prev_results_train = {'lr-LARS/pg1': tensor(1.2800), 'lr-LARS/pg1-momentum': tensor(0.9000), 'lr-LARS/pg2': tensor(1.2800), 'lr-LARS/pg2-momentum': tensor(0.9000), 'lr-LARS/pg3': tensor(1.2800), 'lr-LARS/pg3-momentum': tensor(0.9000), 'lr-LARS/pg4': tensor(1.2800), 'lr-LARS/pg4-momentum': tensor(0.9000), 'train/loss': tensor(2.7390), 'train/loss_step': tensor(2.7390), 'train/loss_epoch': tensor(2.7390), 'train/mIoU': tensor(0.0167), 'val/loss': tensor(1.7372), 'val/mIoU': tensor(0.0337), 'test/loss': tensor(1.7372), 'test/mIoU': tensor(0.0337)}
    cfg = cfg_supervised_segmentation
    with open_dict(cfg['pretrain']):
        cfg['pretrain']['model']['trainer']['deterministic'] = True
        cfg['pretrain']['model']['trainer']['precision'] = 32
        # cfg['pretrain']['model']['dataloaders']['batch_size'] = 32
    seetests.pretrain_eval_finetune_test(cfg, prev_results_train, accelerator='gpu', check_keys_train=False) # can't do gpu as non-deterministic


# pytest -s tests/test_train.py::test_dinov2_linear_finetune_22_fast_dev_run_gpu
@RunIf(min_gpus=1)
def test_dinov2_linear_finetune_22_fast_dev_run_gpu(cfg_dinov2_linear_finetune_224):
    prev_results_train = {'lr-SGD': tensor(0.0001), 'lr-SGD-momentum': tensor(0.9000), 'train/loss': tensor(2.2950), 'train/loss_step': tensor(2.2950), 'train/loss_epoch': tensor(2.2950), 'train/accuracy': tensor(0.1406), 'train/mean_class_accuracy': tensor(0.1246), 'val/loss': tensor(1.8311), 'val/accuracy': tensor(0.3047), 'val/mean_class_accuracy': tensor(0.0508), 'val/aucroc': tensor(0.), 'val/average_precision_0': tensor(1.), 'val/average_precision_1': tensor(nan), 'val/average_precision_2': tensor(nan), 'val/average_precision_3': tensor(nan), 'val/average_precision_4': tensor(nan), 'val/average_precision_5': tensor(nan), 'val/average_precision_6': tensor(nan), 'val/average_precision_7': tensor(nan), 'val/average_precision_8': tensor(nan), 'val/average_precision_9': tensor(nan), 'val/max_acc': tensor(0.3047), 'test/loss': tensor(1.8311), 'test/accuracy': tensor(0.3047), 'test/mean_class_accuracy': tensor(0.0508), 'test/aucroc': tensor(0.), 'test/average_precision_0': tensor(1.), 'test/average_precision_1': tensor(nan), 'test/average_precision_2': tensor(nan), 'test/average_precision_3': tensor(nan), 'test/average_precision_4': tensor(nan), 'test/average_precision_5': tensor(nan), 'test/average_precision_6': tensor(nan), 'test/average_precision_7': tensor(nan), 'test/average_precision_8': tensor(nan), 'test/average_precision_9': tensor(nan)}
    cfg = cfg_dinov2_linear_finetune_224
    with open_dict(cfg['pretrain']):
        cfg['pretrain']['model']['methods']['learning_rates'] = [0.001]
        cfg['pretrain']['model']['dataloaders']['batch_size'] = 128
        cfg['pretrain']['model']['trainer']['precision'] = 32
        cfg['pretrain']['model']['trainer']['deterministic'] = False
    seetests.pretrain_eval_finetune_test(cfg, prev_results_train, accelerator='gpu', check_keys_train=False) # can't do gpu as non-deterministic

