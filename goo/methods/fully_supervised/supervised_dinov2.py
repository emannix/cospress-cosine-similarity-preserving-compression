
from ..model_base_s import ModelBaseS
from pdb import set_trace as pb

from ...networks.dinov2_linear_classifier import create_linear_input

from torchmetrics.functional.classification import multiclass_accuracy
from torch.nn.functional import softmax, normalize
import torch.nn as nn
import torch

import torch.distributed as dist
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List

from enum import Enum
from torchmetrics import Metric, MetricCollection
from torchmetrics.classification import MulticlassAccuracy
from torchmetrics import AUROC, AveragePrecision
class MetricType(Enum):
    MEAN_ACCURACY = "mean_accuracy"
    MEAN_PER_CLASS_ACCURACY = "mean_per_class_accuracy"
    PER_CLASS_ACCURACY = "per_class_accuracy"
    IMAGENET_REAL_ACCURACY = "imagenet_real_accuracy"

    @property
    def accuracy_averaging(self):
        return getattr(AccuracyAveraging, self.name, None)

    def __str__(self):
        return self.value
class AccuracyAveraging(Enum):
    MEAN_ACCURACY = "micro"
    MEAN_PER_CLASS_ACCURACY = "macro"
    PER_CLASS_ACCURACY = "none"

    def __str__(self):
        return self.value

def is_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_global_size() -> int:
    return dist.get_world_size() if is_enabled() else 1

def get_global_rank() -> int:
    return dist.get_rank() if is_enabled() else 0

def scale_lr(learning_rates, batch_size):
    return learning_rates * (batch_size * get_global_size()) / 256.0

class AllClassifiers(nn.Module):
    def __init__(self, classifiers_dict):
        super().__init__()
        self.classifiers_dict = nn.ModuleDict()
        self.classifiers_dict.update(classifiers_dict)

    def forward(self, inputs):
        return {k: v.forward(inputs) for k, v in self.classifiers_dict.items()}

    def __len__(self):
        return len(self.classifiers_dict)

class Supervised(ModelBaseS):
    def __init__(self, 
        learning_rates = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 0.1], 
        avgpool = [True, False], 
        n_last_blocks = [1,4],
        batch_size = 128,
        accuracy_type = 'MEAN_ACCURACY',
        force_num_classes = None,
        normalize_backbone_outputs = False,
        freeze_backbone = False,
        **kwargs):
        super(Supervised, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        
        self.metrics = ['index', 'label', 'y_hat']
        self.metrics_log = [False, False, False]
        self.metrics_save = [True, True, True]

        # torch.set_default_dtype(torch.float32)
        # torch.manual_seed(0)
        # sample = torch.rand((1, 3, 224, 224))
        # sample = sample.cuda()
        # model = self.networks.backbone.keywords['feature_model']
        # model.cuda()
        # model = model.float()
        # model(sample)

        n_last_blocks = max(n_last_blocks)
        self.networks.backbone.keywords['n_last_blocks'] = n_last_blocks
        self.backbone = self.networks.backbone().cuda()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False   

        test_init = self.networks.discriminator()
        if isinstance(test_init, AllClassifiers):
            self.discriminator = test_init
        else:
            sample = torch.rand((2, 3, 224, 224)).cuda()
            sample_output = self.forward(sample, discriminator=False)
            # sample_output = self.backbone(sample)

            # if not isinstance(sample_output, tuple):
            #     shp = len(sample_output.shape)
            #     if shp == 2:
            #         sample_output = ((None, sample_output),)
            #     elif shp == 3:
            #         sample_output = ((sample_output[:,1:,:], sample_output[:,0,:]),)


            linear_classifiers_dict = nn.ModuleDict()
            optim_param_groups = []
            for n in self.hparams.n_last_blocks:
                for avgpool in self.hparams.avgpool:
                    for _lr in self.hparams.learning_rates:
                        lr = scale_lr(_lr, batch_size)
                        out_dim = create_linear_input(sample_output, use_n_blocks=n, use_avgpool=avgpool).shape[1]
                        
                        self.networks.discriminator.keywords['out_dim'] = out_dim
                        self.networks.discriminator.keywords['use_avgpool'] = avgpool
                        self.networks.discriminator.keywords['use_n_blocks'] = n
                        if force_num_classes is not None:
                            self.networks.discriminator.keywords['output_dim'] = force_num_classes
                        linear_classifier = self.networks.discriminator()
                        linear_classifiers_dict[
                            f"classifier_{n}_blocks_avgpool_{avgpool}_lr_{lr:.6f}".replace(".", "_")
                        ] = linear_classifier
                        optim_param_groups.append({"params": linear_classifier.parameters(), "lr": lr})

            linear_classifiers = AllClassifiers(linear_classifiers_dict)
            self.discriminator = linear_classifiers

        self.register_buffer("best_classifier", torch.tensor([0]))
        # self.best_classifier_module = None
        self.best_classifier_module = linear_classifier.linear
        self.set_best_classifier('none')

        self.optim_param_groups = optim_param_groups 
        # ====================================
        self.accuracy_type = accuracy_type
        if accuracy_type == 'average_precision':
            metrics = {
                f"top-1": AveragePrecision(task='binary', num_classes=self.num_classes)
            }
        else:
            ks = [1]
            average_type = getattr(MetricType, self.accuracy_type).accuracy_averaging.value
            metrics = {
                f"top-{k}": MulticlassAccuracy(top_k=k, num_classes=self.num_classes, average=average_type) for k in ks
            }
        self.head_metrics = {k: MetricCollection(metrics).clone() for k in self.discriminator.classifiers_dict}

    # ================================================
    def set_best_classifier(self, k):
        ords = list(map(ord, k))
        self.best_classifier = torch.tensor(ords)
    def get_best_classifier(self):
        ords = self.best_classifier.tolist()
        return "".join(map(chr, ords))

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        self.best_classifier = state_dict['best_classifier']
        strict = False
        super(Supervised, self).load_state_dict(state_dict, strict)

        self.best_classifier_module = self.discriminator.classifiers_dict[self.get_best_classifier()].linear
    # ================================================

    def configure_optimizers(self):
        configured = super(Supervised, self).configure_optimizers()

        for i in range(len(configured['optimizer'].param_groups)):
            del configured['optimizer'].param_groups[0]

        for group in self.optim_param_groups:
            configured['optimizer'].add_param_group(group)
        return configured

    def forward(self, x, discriminator = True):
        z = self.backbone(x)
        if not isinstance(z, tuple):
            shp = len(z.shape)
            if shp == 2:
                z = ((None, z),)
            elif shp == 3:
                z = ((z[:,1:,:], z[:,0,:]),)

        if False:
        # This gets similar results - the up-weighting on the final line is important for some reason (temp of 0.1)
            if not isinstance(z, tuple):
                shp = len(z.shape)
                if shp == 2:
                    z = [[None, z],]
                elif shp == 3:
                    z = [[z[:,1:,:], z[:,0,:]],]

            if self.hparams.normalize_backbone_outputs:
                for i in range(len(z[0])):
                    if z[0][i] is not None:
                        z[0][i] = normalize(z[0][i], dim=-1)*12
        # pb()

        if discriminator:
            y_hat = self.discriminator(z)
        else:
            y_hat = z
        return y_hat

    def model_step(self, batch, stage='fit'):
        [x], y, idx = batch
        y_hat = self.forward(x)
        losses = {f"{k}": self.loss(v, y) for k, v in y_hat.items()}
        loss = sum(losses.values())
        # pb()
        # =====================================
        if self.get_best_classifier() == 'none':
            # best_classifier = min(losses, key=losses.get)
            best_classifier = next(iter(losses)) # get first one as dummy
        else:
            best_classifier = self.get_best_classifier()
        y_hat_prob = softmax(y_hat[best_classifier], dim=1)
        # acc = multiclass_accuracy(y_hat_prob, y, self.num_classes)
        results_dict = {
            'loss': loss.mean(), 'index': idx, 'label': y, 'y_hat': y_hat_prob #, 'acc': acc
        }

        # for i in range(len(self.trainer.optimizers[0].param_groups)):
        #     for j in range(len(self.trainer.optimizers[0].param_groups[i]['params'])):
        #         if not self.trainer.optimizers[0].param_groups[i]['params'][j].is_cuda:
        #             pb()

        if stage != 'fit':
            for k, metric in self.head_metrics.items():
                metric.to(y_hat[k])
                if self.accuracy_type == 'average_precision':
                    metric.update(softmax(y_hat[k], dim=1)[:, 1], y)
                else:
                    metric.update(softmax(y_hat[k], dim=1), y)
        # =====================================
        return results_dict
        

    def on_validation_epoch_start(self):
        pass
        # pb()
    #     self.trainer.predict_loop._return_predictions: bool = True

    #     datamodule = self.trainer.datamodule
    #     orig_default = datamodule.hparams.predict_dataloader_default
    #     datamodule.hparams.predict_dataloader_default = 'data_val'
    #     self.trainer.reset_predict_dataloader(self)
    #     predictions = self.trainer.predict_loop.run()
    #     datamodule.hparams.predict_dataloader_default = orig_default

    #     pb()
    def on_validation_epoch_end(self):
        stats = {k: metric.compute() for k, metric in self.head_metrics.items()}
        # rank = get_global_rank()
        # if rank == 0:
        max_acc = 0
        for k, metric in self.head_metrics.items():
            metric.reset()
            if stats[k]['top-1'] > max_acc:
                max_acc = stats[k]['top-1']
                self.set_best_classifier(k)
                self.best_classifier = self.best_classifier.to(max_acc.device)

        print(self.get_best_classifier())
        self.log("val/max_acc", max_acc, on_epoch=True, prog_bar=True, sync_dist=True)
        

    def on_test_epoch_start(self):
        print(self.get_best_classifier())

# ===========================================================================

    def on_predict_epoch_start(self):
        print(self.get_best_classifier())
        if self.get_best_classifier() != 'none':
            self.discriminator_eval = self.discriminator.classifiers_dict[self.get_best_classifier()]
            self.discriminator_eval.register_forward_hook(self.discriminator._forward_hooks[4])
        else:
            self.discriminator_eval = self.discriminator

    def predict_step(self, batch: Any, batch_idx: int):
        [x], y, idx = batch
        z = self.backbone(x)
        y_hat = self.discriminator_eval(z)
        return y_hat
