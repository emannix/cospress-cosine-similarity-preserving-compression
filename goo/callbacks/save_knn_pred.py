import torchmetrics
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from lightning.pytorch.callbacks import Callback
from ..lightly.dataset import LightlyDataset
from ..datasets.helper import MySubset

from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import copy

from pdb import set_trace as pb
from tqdm import tqdm

import os
from pathlib import Path

from torchmetrics import JaccardIndex 
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification import MulticlassAccuracy

def knn_predict(
    feature: torch.Tensor,
    feature_bank: torch.Tensor,
    feature_labels: torch.Tensor,
    num_classes: int,
    knn_k: int = 200,
    knn_t: float = 0.1,
) -> torch.Tensor:
    """Run kNN predictions on features based on a feature bank
    This method is commonly used to monitor performance of self-supervised
    learning methods.
    The default parameters are the ones
    used in https://arxiv.org/pdf/1805.01978v1.pdf.
    Args:
        feature:
            Tensor of shape [N, D] for which you want predictions
        feature_bank:
            Tensor of a database of features used for kNN
        feature_labels:
            Labels for the features in our feature_bank
        num_classes:
            Number of classes (e.g. `10` for CIFAR-10)
        knn_k:
            Number of k neighbors used for kNN
        knn_t:
            Temperature parameter to reweights similarities for kNN
    Returns:
        A tensor containing the kNN predictions
    Examples:
        >>> images, targets, _ = batch
        >>> feature = backbone(images).squeeze()
        >>> # we recommend to normalize the features
        >>> feature = F.normalize(feature, dim=1)
        >>> pred_labels = knn_predict(
        >>>     feature,
        >>>     feature_bank,
        >>>     targets_bank,
        >>>     num_classes=10,
        >>> )
    """

    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(
        feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices
    )
    # we do a reweighting of the similarities
    sim_weight = (sim_weight / knn_t).exp()
    # counts for each class
    one_hot_label = torch.zeros(
        feature.size(0) * knn_k, num_classes, device=sim_labels.device
    )
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(
        dim=-1, index=sim_labels.view(-1, 1), value=1.0
    )
    # weighted score ---> [B, C]
    pred_scores = torch.sum(
        one_hot_label.view(feature.size(0), -1, num_classes)
        * sim_weight.unsqueeze(dim=-1),
        dim=1,
    )
    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

# ==================================================================
# ==================================================================

class KNNPrediction(Callback):

    def __init__(self, backbone='backbone', check_knn_every: int = 10,
            nearest_neighbours: int = 200, temperature: float = 0.1, 
            sample_frac: float = 1.0, ignore_index = None,
            save_figures=False, trim_unlabelled=False, shuffle_knn=True, log_mIoU=False):
        super().__init__()
        self.nearest_neighbours = nearest_neighbours
        self.temperature = temperature
        self.sample_frac = sample_frac
        self.backbone_str = backbone
        self.check_every = check_knn_every
        self.dp = nn.Parameter(torch.tensor(0.0))
        self.reported = True
        self.log_mIoU = log_mIoU
        self.save_figures = save_figures
        self.trim_unlabelled = trim_unlabelled
        self.shuffle_knn = shuffle_knn
        self.ignore_index = ignore_index

    def setup(self, trainer, pl_module, stage):
        self.backbone = getattr(pl_module, self.backbone_str)
        self.num_classes = trainer.datamodule.num_classes

        self.miniters = trainer.log_every_n_steps
        if isinstance(trainer.limit_val_batches, int):
            self.nearest_neighbours = min(trainer.limit_val_batches, self.nearest_neighbours)
            self.val_limit_batches = True
        else:
            self.val_limit_batches = False
        if isinstance(trainer.limit_train_batches, int):
            self.train_limit_batches = True
        else:
            self.train_limit_batches = False

        self.pl_module = pl_module

        self.metric_accuracy = Accuracy(task="multiclass", num_classes=self.num_classes, average='micro')
        self.metric_mean_class_accuracy = MulticlassAccuracy(num_classes=self.num_classes, average='macro')
        self.metric_mIoU = JaccardIndex(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=self.ignore_index)


    def _setup_data(self, trainer, stage='val', sample_indices = None, shuffle_knn=True, basedata='train'):
        datamodule = trainer.datamodule
        transform = datamodule.aug_predict

        trainer.fit_loop.setup_data() # obselete
        dataloader_kNN = datamodule._predict_dataloader()
        if basedata == 'train':
            dataset = copy.deepcopy(datamodule.data_train)
        elif basedata == 'train_labelled':
            dataset = copy.deepcopy(datamodule.data_train_labelled)

        if sample_indices is not None: # and stage != 'test'
            dataset = MySubset(dataset, sample_indices)

        dataset = LightlyDataset.from_torch_dataset(dataset, transform, datamodule.aug_targets)
        dataloader_kNN.keywords['shuffle'] = shuffle_knn # obselete
        dataloader_kNN.keywords['drop_last'] = False # obselete
        dataloader_kNN = datamodule.predict_dataloader(dataset=dataset, base_dataloader=dataloader_kNN)

        if stage is not None:
            if stage == 'val':
                trainer.validate_loop.setup_data() # obselete
                dataloader_pred = copy.deepcopy(datamodule.val_dataloader())
            elif stage == 'test':
                trainer.test_loop.setup_data() # obselete
                dataloader_pred = copy.deepcopy(datamodule.test_dataloader())
            elif stage == 'train':
                dataloader_pred = copy.deepcopy(datamodule.predict_dataloader())
            elif stage == 'train_unlabelled':
                dataloader_pred = copy.deepcopy(datamodule.predict_dataloader(dataset=datamodule.data_train_unlabelled))
            dataloader_pred.dataset.transform = transform
        else:
            dataloader_pred = None

        return dataloader_kNN, dataloader_pred

    def setup_data(self, trainer, stage='val', basedata='train'):
        dataset = trainer.datamodule.data_train
        if self.sample_frac < 1.0: # and stage != 'test'
            sample_indices = np.random.randint(0, len(dataset), int(self.sample_frac*len(dataset)))
        else:
            sample_indices = None
        return self._setup_data(trainer, stage=stage, sample_indices=sample_indices, basedata=basedata,
                shuffle_knn = self.shuffle_knn)

    def _evaluate_embeddings(self, train_data, trainer, rank, nonormalize=False):
        # ==================================================
        self.backbone.eval()
        self.dp = self.dp.to(next(self.backbone.parameters()).device)
        # https://github.com/Lightning-AI/lightning/issues/10430
        # ==================================================
        train_feature_bank = []
        train_targets_bank = []
        with torch.no_grad():
            for i,data in enumerate(tqdm(train_data, position=rank, miniters=self.miniters)):
                [img], target, _ = data
                # if len(target.shape) > 1:
                #     target = target.reshape(-1)
                #     target = target.long()
                img, target = img.to(self.dp.device), target.to(self.dp.device)
                feature = self.backbone(img).squeeze()
                if not nonormalize:
                    feature = F.normalize(feature, dim=-1)
                train_feature_bank.append(feature)
                train_targets_bank.append(target)
                if self.train_limit_batches:
                    if i == trainer.limit_train_batches:
                        break

        train_feature_bank = torch.cat(train_feature_bank, dim=0).contiguous()
        train_targets_bank = torch.cat(train_targets_bank, dim=0).contiguous()
        if dist.is_initialized() and dist.get_world_size() > 1:
            train_feature_bank_all = [torch.zeros_like(train_feature_bank) for _ in range(dist.get_world_size())]
            train_targets_bank_all = [torch.zeros_like(train_targets_bank) for _ in range(dist.get_world_size())]
            dist.all_gather(train_feature_bank_all, train_feature_bank)
            dist.all_gather(train_targets_bank_all, train_targets_bank)
            train_feature_bank_all = torch.cat(train_feature_bank_all, dim=0)
            train_targets_bank_all = torch.cat(train_targets_bank_all, dim=0)
        else:
            train_feature_bank_all = train_feature_bank
            train_targets_bank_all = train_targets_bank

        if self.ignore_index is not None:
            keep_indices = train_targets_bank_all != self.ignore_index
            train_feature_bank_all = train_feature_bank_all[keep_indices, :]
            train_targets_bank_all = train_targets_bank_all[keep_indices]

        if self.trim_unlabelled:
            train_feature_bank_all = train_feature_bank_all[train_targets_bank_all != -1, :]
            train_targets_bank_all = train_targets_bank_all[train_targets_bank_all != -1]

        if self.save_figures:
            from cuml.manifold import TSNE
            import matplotlib.pyplot as plt
            tsne_params={'n_components':2, 'metric':'cosine'}
            tsne = TSNE(**tsne_params)
            tsne_cuml = tsne.fit_transform(train_feature_bank_all.t())
            tsne_cuml = tsne_cuml.get()
            plt.figure()
            plt.scatter(tsne_cuml[:,0], tsne_cuml[:,1], c=train_targets_bank_all.cpu(), s = 0.5)
            plt.tight_layout()
            logger = trainer.logger
            output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'knn_tsne')
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_file_base = 'epoch='+str(trainer.current_epoch)+'-step='+str(trainer.global_step)+'-tsne.png'
            plt.savefig(output_folder + '/' +output_file_base)

        if len(train_feature_bank.shape) == 2:
            train_feature_bank_all = train_feature_bank_all.t()
            train_targets_bank_all = train_targets_bank_all.t()

        return train_feature_bank_all, train_targets_bank_all

    def evaluate(self, trainer, pl_module, stage='val'):
        train_data, test_data = self.setup_data(trainer, stage)
        if dist.is_initialized() and dist.get_world_size() > 1:
            rank = dist.get_rank()
        else:
            rank = 0
        train_feature_bank_all, train_targets_bank_all = self._evaluate_embeddings(train_data, trainer, rank)
        # ====================================================
        num = torch.zeros([self.num_classes]).to(self.dp.device)
        top1 = torch.zeros([self.num_classes]).to(self.dp.device)
        with torch.no_grad():
            for i,data in enumerate(tqdm(test_data, position=rank, miniters=self.miniters)):
                [img], target, _ = data
                if len(target.shape) > 1:
                    target = target.reshape(-1)
                    target = target.long()
                img, target = img.to(self.dp.device), target.to(self.dp.device)
                feature = self.backbone(img).squeeze()
                feature = F.normalize(feature, dim=1)
                pred_labels = knn_predict(
                    feature,
                    train_feature_bank_all,
                    train_targets_bank_all,
                    self.num_classes,
                    self.nearest_neighbours,
                    self.temperature,
                )

                self.metric_accuracy.to(feature).update(pred_labels[:, 0], target)
                self.metric_mean_class_accuracy.to(feature).update(pred_labels[:, 0], target)
                if self.log_mIoU:
                    self.metric_mIoU.to(feature).update(pred_labels[:, 0], target)

                if self.val_limit_batches:
                    if i == trainer.limit_val_batches:
                        break

        self.report_acc = self.metric_accuracy.compute()
        self.report_acc_mean = self.metric_mean_class_accuracy.compute()
        # self.report_acc_mean = torch.tensor([0.0])
        if self.log_mIoU:
            self.report_mIoU = self.metric_mIoU.compute()
        self.reported = False

        self.metric_accuracy.reset()
        self.metric_mean_class_accuracy.reset()
        if self.log_mIoU:
            self.metric_mIoU.reset()

# ========================================================
    # def on_train_start(self, trainer, pl_module):
    #     self.evaluate_knn(trainer, pl_module, stage='val')
    def _on_epoch_end(self, trainer, pl_module, string='val/kNN_accuracy'):
        device = self.pl_module.device
        if not self.reported:
            self.log(string, self.report_acc.to(device), prog_bar=True, sync_dist=True)
            self.log(string+'_mean', self.report_acc_mean.to(device), prog_bar=True, sync_dist=True)
            if self.log_mIoU:
                self.log(string+'_mIoU', self.report_mIoU.to(device), prog_bar=True, sync_dist=True)
            self.reported = True

    def on_validation_start(self, trainer, pl_module):
        if trainer.current_epoch % self.check_every == self.check_every-1:
            self.evaluate(trainer, pl_module, stage='val')
    def on_validation_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, string='val/'+self.backbone_str+'_kNN_accuracy')

    def on_test_start(self, trainer, pl_module):
        self.evaluate(trainer, pl_module, stage='test')
    def on_test_epoch_end(self, trainer, pl_module):
        self._on_epoch_end(trainer, pl_module, string='test/'+self.backbone_str+'_kNN_accuracy')
