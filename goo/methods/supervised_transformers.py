
from .model_base_s import ModelBaseS
from pdb import set_trace as pb

from torchmetrics.functional.classification import multiclass_accuracy
from torch.nn.functional import softmax
# Based on https://github.com/facebookresearch/deit
from timm.data import Mixup
from timm.utils import ModelEma
import torch
import numpy as np
import random

class SupervisedTransformers(ModelBaseS):
    def __init__(self,
        mixup = 0.8, cutmix=1.0, cutmix_minmax=None, mixup_prob=1.0,
        mixup_switch_prob=0.5, mixup_mode='batch', smoothing=0.1,
        model_ema = 0.0, mixup_seed = None, freeze_backbone=False,
        **kwargs):
        super(SupervisedTransformers, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        
        self.metrics = ['index', 'label', 'y_hat']
        self.metrics_log = [False, False, False]
        self.metrics_save = [True, True, True]

        self.backbone = self.networks.backbone()
        self.discriminator = self.networks.discriminator()

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        if model_ema > 0:
            self.model_ema = ModelEma(self, decay=self.hparams.model_ema)
        else:
            self.model_ema = None

        if mixup > 0 or cutmix > 0 or cutmix_minmax is not None:
            self.mixup_fn = Mixup(
            mixup_alpha=mixup, cutmix_alpha=cutmix, cutmix_minmax=cutmix_minmax,
            prob=mixup_prob, switch_prob=mixup_switch_prob, mode=mixup_mode,
            label_smoothing=smoothing, num_classes=kwargs['num_classes'])
        else:
            self.mixup_fn = None

        self.loss_eval = torch.nn.CrossEntropyLoss()
        # ==============================================
        # self.automatic_optimization = False
        # self.scaler = torch.cuda.amp.GradScaler()

    def forward(self, x):
        z = self.backbone(x)
        if len(z.shape) > 2:
            z = z[:, 0, :] # select just class token
        # return z
        y_hat = self.discriminator(z)
        return y_hat

    def model_step(self, batch, stage='fit'):
        [x], y, idx = batch
        if self.hparams.mixup_seed is not None:
            torch.set_printoptions(10)
            torch.manual_seed(self.hparams.mixup_seed)
            np.random.seed(self.hparams.mixup_seed)
            random.seed(self.hparams.mixup_seed)
        if self.mixup_fn is not None and stage == 'fit':
            x, y_mixed = self.mixup_fn(x, y)
        else:
            y_mixed = y

        y_hat = self.forward(x)
        if stage == 'fit':
            loss = self.loss(y_hat, y_mixed)
        else:
            loss = self.loss_eval(y_hat, y_mixed)

        if self.hparams.mixup_seed is not None:
            print(loss)
            pb()
        # =====================================
        y_hat_prob = softmax(y_hat, dim=1)
        # acc = multiclass_accuracy(y_hat_prob, y, self.num_classes)
        results_dict = {
            'loss': loss.mean(), 'index': idx, 'label': y, 'y_hat': y_hat_prob 
        }
        # =====================================
        # if stage == 'fit':
        #     optimizer = self.optimizers()
        #     optimizer.zero_grad()
        #     if loss.device.type != 'cpu':
        #         # scaler = self.trainer.precision_plugin.scaler
        #         scaler = self.scaler
        #         loss = scaler.scale(loss)
        #         loss.backward(create_graph=False)
        #         scaler.step(optimizer)
        #         scaler.update()
        #     else:
        #         loss.backward(create_graph=False)
        #         optimizer.step()
        # =====================================
        return results_dict

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # do something on_after_optimizer_step
        torch.cuda.synchronize()
        if self.model_ema is not None:
            self.model_ema.update(self)


    # =====================================
    
    def predict_step(self, batch, batch_idx):
        [x], y, idx = batch
        res = self.backbone(x)
        return res, idx
