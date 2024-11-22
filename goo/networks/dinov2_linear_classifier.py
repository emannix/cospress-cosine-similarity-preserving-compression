
import argparse
from functools import partial
import json
import logging
import os
import sys
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
from pdb import set_trace as pb

def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else t.squeeze()
def to_tuple(lst):
    return tuple(to_tuple(i) if isinstance(i, list) else i for i in lst)

def create_linear_input(x_tokens_list, use_n_blocks, use_avgpool):
    intermediate_output = x_tokens_list[-use_n_blocks:]
    output = torch.cat([class_token for _, class_token in intermediate_output], dim=-1)
    if use_avgpool:
        output = torch.cat(
            (
                output,
                torch.mean(intermediate_output[-1][0], dim=1),  # patch tokens
            ),
            dim=-1,
        )
        output = output.reshape(output.shape[0], -1)
    return output.float()


class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""

    def __init__(self, out_dim, use_n_blocks, use_avgpool, output_dim=1000, seed=None, normalize=None):
        super().__init__()
        num_classes = output_dim
        self.out_dim = out_dim
        self.use_n_blocks = use_n_blocks
        self.use_avgpool = use_avgpool
        self.num_classes = num_classes
        if seed is not None:
            torch.manual_seed(seed)
        self.linear = nn.Linear(out_dim, num_classes)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()
        if normalize is None:
            self.linear_forward = self.linear
        elif normalize == 'layernorm':
            self.linear_forward = nn.Sequential(
              nn.LayerNorm(out_dim),
              self.linear)
        elif normalize == 'batchnorm':
            self.linear_forward = nn.Sequential(
              torch.nn.BatchNorm1d(out_dim, affine=False),
              self.linear)

    def forward(self, x_tokens_list):
        output = create_linear_input(x_tokens_list, self.use_n_blocks, self.use_avgpool)
        return self.linear_forward(output)

# ==============================================

class ModelWithIntermediateLayers(nn.Module):
    def __init__(self, feature_model, n_last_blocks, 
            outputs='all', no_inference=False, train_feature_model=False, inner_model=False,
            timm_interface = False, **kwargs):
        super().__init__()
        
        if inner_model:
            self.feature_model = feature_model.model
        else:
            self.feature_model = feature_model

        self.timm_interface = timm_interface
        self.train_feature_model = train_feature_model
        if train_feature_model == False:
            self.feature_model.eval()
        self.n_last_blocks = n_last_blocks
        autocast_ctx = partial(torch.cuda.amp.autocast, enabled=True, dtype=torch.half)
        self.autocast_ctx = autocast_ctx
        self.outputs = outputs
        self.no_inference = no_inference

        if self.train_feature_model == False:
            for param in self.feature_model.parameters():
                param.requires_grad = False

        # self.feature_model.cuda()
        # torch.manual_seed(0)
        # sample = torch.rand((1, 3, 224, 224)).cuda()
        # self.feature_model(sample)

    def forward(self, images, n_last_blocks=None):
        if n_last_blocks is None:
            n_last_blocks = self.n_last_blocks
        if self.timm_interface:
            if self.train_feature_model == False:
                self.feature_model.eval()
                with torch.inference_mode():
                    with self.autocast_ctx():
                        features = self.feature_model.get_intermediate_layers(
                            images, n_last_blocks, return_class_token=True, norm=True
                        )
                        features = listit(features)
                        if len(features[0][1].shape) > 2: # dropping distillation token
                            for i in range(len(features)):
                                features[i][1] = features[i][1][:,0,:]
                                
                        features = to_tuple(features)
        else:
            if self.train_feature_model == False:
                self.feature_model.eval()
                with torch.inference_mode():
                    with self.autocast_ctx():
                        features = self.feature_model.get_intermediate_layers(
                            images, n_last_blocks, return_class_token=True
                        )
            else:
                with self.autocast_ctx():
                    features = self.feature_model.get_intermediate_layers(
                            images, n_last_blocks, return_class_token=True
                        )

        if self.outputs == 'all':
            features = features
        elif self.outputs == 'cls_tkn':
            features = features[0][1]
        elif self.outputs == 'patch_tkn':
            features = torch.stack([features[i][0] for i in range(len(features))]).squeeze()
        elif self.outputs == 'patch_tkn_noimg':
            features = features[0][0]
            features = features.reshape(-1, features.shape[2])
        elif self.outputs == 'all_combined':
            features = torch.stack([torch.concatenate([features[i][1].unsqueeze(1), features[i][0]], dim=1) for i in range(len(features))]).squeeze()
            
        if self.no_inference:
            if features.is_inference():
                features = torch.tensor(features).to(images)
        return features

    def forward_patch_embed(self, images):
        if self.train_feature_model == False:
            self.feature_model.eval()
            with torch.inference_mode():
                with self.autocast_ctx():
                    features = self.feature_model.patch_embed(images)
        else:
            with self.autocast_ctx():
                features = self.feature_model.patch_embed(images)

        if self.no_inference:
            if features.is_inference():
                features = torch.tensor(features).to(images)
        return features

    def forward_to_n(self, x: torch.Tensor, n: int) -> torch.Tensor:
        model = self.feature_model
        model.eval()
        with torch.inference_mode():
            with self.autocast_ctx():
                x = model.prepare_tokens_with_masks(x)
                blocks = model.blocks[:(n+1)]
                if model.chunked_blocks:
                    i = 0
                    for block_chunk in blocks:
                        for blk in block_chunk[i:]:  # Passing the nn.Identity()
                            x = blk(x)
                            i += 1
                else:
                    for i, blk in enumerate(blocks):
                        x = blk(x)
                x = model.norm(x)
                
        if self.outputs == 'patch_tkn':
            return x[:,1:]
        else:
            return x