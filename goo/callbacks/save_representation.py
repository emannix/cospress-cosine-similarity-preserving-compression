import torchmetrics
import torch
from torchmetrics import Metric
from torchmetrics.utilities.data import dim_zero_cat

from lightning.pytorch.callbacks import Callback

from typing import Any, Callable, Dict, Optional, Union
from pathlib import Path
import os
import pandas as pd
import glob
from operator import attrgetter

from torchmetrics import AUROC, AveragePrecision
import torch
import numpy as np

from pdb import set_trace as pb

class RepresentationCache(Metric):
    full_state_update=True
    def __init__(self, dist_sync_on_step=False, 
            compress_arrays=False,
            only_class_token=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.record = False
        self.representation = []
        self.compress_arrays = compress_arrays
        self.only_class_token = only_class_token

    def hook(self, model, input, output):
        self.update(output)

    def update(self, representation: torch.Tensor):
        if self.record:
            if isinstance(representation, tuple):
                res = [x.detach().cpu() for sublist in representation for x in sublist]
                if self.only_class_token:
                    res = res[1]
            else:
                res = representation.detach().cpu()
            if self.compress_arrays:
                if isinstance(res, list):
                    res = [x.half() for x in res]
                else:
                    res = res.half()
            self.representation.append(res)

    def reset(self):
        self.representation = []
        
    def return_results(self):
        if (len(self.representation) > 0):
            if isinstance(self.representation[0], list):
                representation = []
                for i in range(len(self.representation[0])):
                    representation.append(dim_zero_cat([x[i] for x in self.representation]).squeeze())
            else:
                representation = dim_zero_cat(self.representation).squeeze()
        else:
            representation = torch.tensor((0.0, 0.0))

        return(representation)
    
    def compute(self):
        pass

# ==================================================================
# ==================================================================

class SaveRepresentation(Callback):

    def __init__(self, layers=['encoder', 'encoder_mapping'], 
            stages=['predict'],
            compress_arrays=False,
            target_dataset='',
            only_class_token=True):
        super().__init__()
        self.layers = layers
        self.stages = stages
        self.target_dataset = target_dataset
        self.labels = None
        self.representation_cache = {}
        for key in self.layers:
            self.representation_cache[key] = RepresentationCache(
                    compress_arrays=compress_arrays, only_class_token=only_class_token)

    def setup(self, trainer, pl_module, stage):
        if self.target_dataset != '':
            self.labels = getattr(trainer.datamodule, self.target_dataset).targets
        for key, cache in self.representation_cache.items():
            retriever = attrgetter(key)
            retriever(pl_module).register_forward_hook(cache.hook)

    def get_output_folder(self, trainer):
        logger = trainer.logger
        output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'predictions')
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        output_file_base = 'epoch='+str(trainer.current_epoch)+'-step='+str(trainer.global_step)+'-'
        return output_folder, output_file_base

    def save_labels_as_csv(self, trainer):
        if self.labels is not None:
            output_folder, output_file_base = self.get_output_folder(trainer)
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
            if rank == 0:
                output_file_final = output_folder + '/' + self.target_dataset +'_labels'
                np.array(self.labels).dump(output_file_final + '.npy')

    def save_predictions_as_csv(self, cache, trainer, key, save_string):
        print('saving results')
        representation = cache.return_results()
        output_folder, output_file_base = self.get_output_folder(trainer)

        if torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0
        output_file_final = output_folder + '/' + output_file_base + save_string +key+'_' +str(rank)

        if isinstance(representation, list):
            for i in range(len(representation)):
                rep = representation[i].cpu().numpy()
                rep.dump(output_file_final + '_' + str(i) + '.npy')
        else:
            representation = representation.cpu().numpy()
            representation.dump(output_file_final + '.npy')
        print(output_file_final)
        # representation_values = pd.DataFrame(representation, columns=None)
        # representation_values.to_csv(output_file_final + '.csv', index=False)

    # ==========================================================
    def _epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        for key, value in self.representation_cache.items():
            value.record = True

    def on_predict_epoch_start(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if 'predict' in self.stages:
            self._epoch_start(trainer, pl_module)

    def on_test_epoch_start(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if 'test' in self.stages:
            self._epoch_start(trainer, pl_module)
    # ==========================================================
    def _epoch_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule', save_string='_test_'):
        self.save_labels_as_csv(trainer)
        for key, value in self.representation_cache.items():
            self.save_predictions_as_csv(value, trainer, key, save_string)
            value.reset()
            value.record = False

    def on_predict_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if 'predict' in self.stages:
            self._epoch_end(trainer, pl_module, '_predict_')

    def on_test_epoch_end(
        self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'
    ):
        if 'test' in self.stages:
            self._epoch_end(trainer, pl_module, '_test_')