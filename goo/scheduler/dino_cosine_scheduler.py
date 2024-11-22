# https://github.com/facebookresearch/suncet/blob/main/src/utils.py
import os
import math
import torch
import torch.distributed as dist
import numpy as np
from pdb import set_trace as pb

from .base_scheduler import Scheduler

class CosineScheduler(Scheduler):

    def __init__(
        self,
        optimizer,
        last_epoch=-1,
        T_max = 100,
        warmup_prop = [0.0, 0.0],
        warmup_value = [0, 0],
        base_value = [0, 1],
        final_value = [1, 1],
        attribute = ['lr', 'weight_decay'],
        dino_orig_style_warmup = False,
    ):
        self.dino_orig_style_warmup = dino_orig_style_warmup
        self.warmup_value = np.array(warmup_value)
        self.base_value = np.array(base_value)
        self.final_value = np.array(final_value)
        self.attribute = attribute
        self.T_max = T_max
        self.warmup_prop = warmup_prop
        self.warmup_steps = np.array([int(x*T_max) for x in warmup_prop])

        self.init_param_groups = []
        for i in range(len(optimizer.param_groups)):
            self.init_param_groups.append({
                attr: optimizer.param_groups[i][attr] if attr in optimizer.param_groups[i] else 1.0
                for attr in attribute})

        super(CosineScheduler, self).__init__(
            optimizer,
            last_epoch=last_epoch)

    def compute_warmup(self, step, i):
        if self.dino_orig_style_warmup:
            return self.warmup_value[i] + np.abs(self.base_value[i] - self.warmup_value[i]) * step/(self.warmup_steps[i]-1)
        else:
            if np.sum(self.warmup_value[i]) < 1e-10:
                return self.base_value[i] * (step+1)/(self.warmup_steps[i])
            return self.warmup_value[i] + np.abs(self.base_value[i] - self.warmup_value[i]) * step/(self.warmup_steps[i])

    def compute_cosine(self, step, i):
        if self.base_value[i] < self.final_value[i]:
            sign = -1
        else:
            sign = 1

        return self.final_value[i] + sign*0.5 * np.abs(self.base_value[i] - self.final_value[i]) * \
            (1 + np.cos(np.pi * (step-self.warmup_steps[i]) / (self.T_max - self.warmup_steps[i])))

    def get_lr(self):
        res = []
        for i in range(len(self.attribute)):
            if self.last_epoch < self.warmup_steps[i]:
                resi = self.compute_warmup(self.last_epoch, i)
            else:
                resi = self.compute_cosine(self.last_epoch, i)
            res.append(resi)
        res_groups = {}
        for i in range(len(self.attribute)):
            res_groups[self.attribute[i]] = \
                [res[i]*group[self.attribute[i]] for group in self.init_param_groups]
        return res_groups
