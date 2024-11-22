
from ..model_base_s import ModelBaseS
from pdb import set_trace as pb
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed.nn
import torch.distributed as dist
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm
from ...networks import proteus_dinov2 as models_dinov2
from timm.utils import ModelEma

from ...loss.paws import AllGather, AllReduce
import numpy as np

import torchvision
from PIL import Image
import numpy as np

import copy
from ...lightly.dataset import LightlyDataset
import tqdm
import os
from pathlib import Path
import sys

def log_von_mises_fischer_constant_approx(x, order):
    x = torch.tensor(x)
    order2 = torch.tensor(order/2 - 1)
    r = x/order2
    eta = (1+r**2)**0.5+torch.log(r/(1+(1+r**2)**0.5))

    val = 1.0/4.0 * torch.log(1+(x/order2)**2) + order2*torch.log(x) - order2*eta +\
            1.0/2.0 * torch.log(2*np.pi*order2) - order/2.0 * torch.log(2*torch.tensor(np.pi))
    return val

class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
    
    def forward(self, predictions, targets):
        # Implement your custom loss calculation here
        predictions = predictions.reshape(predictions.shape[0], -1)
        targets = targets.reshape(targets.shape[0], -1)
        predictions = F.normalize(predictions, dim=-1)
        targets = F.normalize(targets, dim=-1)

        loss = torch.sum(predictions * targets, dim=-1)
        loss = torch.mean(1- loss)
        return loss

def x2p_torch_parrallel_Q_patches_calcD(query, support, kernel='vmf', per_image = True, LARGE_NUM = 1e8, mask=False):

    if kernel == 'vmf':
        query = torch.nn.functional.normalize(query, dim=-1)
        support = torch.nn.functional.normalize(support, dim=-1)
        if per_image:
            D = torch.bmm(query, support.permute(0,2,1))
        else:
            D = query @ support.T

    if mask:
        if query.shape == support.shape:
            if per_image:
                mask = torch.nn.functional.one_hot(torch.arange(query.shape[1]), query.shape[1]).to(query)
                D = D - mask[None,:,:]*LARGE_NUM
            else:
                mask = torch.nn.functional.one_hot(torch.arange(query.shape[0]), query.shape[0]).to(query)
                D = D - mask[:,:]*LARGE_NUM
        else:
            batch_size = D.shape[0]
            enlarged_batch_size = D.shape[1]
            rank = dist.get_rank()
            labels_oh = torch.arange(batch_size, device=D.device) + rank * batch_size
            mask = torch.nn.functional.one_hot(labels_oh, enlarged_batch_size).to(query)
            D = D - mask[:,:]*LARGE_NUM

    return D

class BackboneWrapper(nn.Module):
    def __init__(self, teacher):
        super(BackboneWrapper, self).__init__()
        self.teacher = teacher

    def forward(self, x):
        with torch.no_grad():
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)

        backbone_cls_view = teacher_backbone_output_dict["x_norm_clstoken"]


        # backbone_cls_view = torch.nn.functional.normalize(backbone_cls_view, dim=1)
        # backbone_cls_view @ backbone_cls_view.T
        # (backbone_cls_view @ backbone_cls_view.T).min()
        # -0.1957
        # (backbone_cls_view @ backbone_cls_view.T).mean()
        # 0.1279
        # pb()

        return backbone_cls_view

class BackboneHeadWrapper(nn.Module):
    def __init__(self, teacher, teacher_head):
        super(BackboneHeadWrapper, self).__init__()
        self.teacher = teacher
        self.teacher_head = teacher_head

    def forward(self, x):
        with torch.no_grad():
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            backbone_view = teacher_backbone_output_dict["x_prenorm"]
            backbone_view = self.teacher.backbone.norm(backbone_view)

        distilled_view = self.teacher_head(backbone_view)
        return distilled_view[:, 0]

def visualize_tensor_PCA(input_tensor, only_calc=False, num_image = 1):
    testme = input_tensor.reshape(-1, input_tensor.shape[-1])
    testme = torch.nn.functional.normalize(testme, dim=1)
    testme_pca = torch.pca_lowrank(testme)[0]
    num_patches = int((testme.shape[0]/num_image)**0.5)
    testme_pca = testme_pca[:,:3].reshape(num_image, num_patches, num_patches, 3)
    mins  = testme_pca.min(dim=0)[0].min(dim=0)[0]
    testme_pca = testme_pca - mins[None, None, :]
    maxs  = testme_pca.max(dim=0)[0].max(dim=0)[0]
    testme_pca = testme_pca/maxs[None, None, :]
    if only_calc:
        return testme_pca.permute(0, 3,1,2)
    else:
        img = torchvision.transforms.ToPILImage()(testme_pca[0].permute(2,0,1))
        return img

class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)

class Proteus(ModelBaseS):
    def __init__(self,
            # ======================
            target_model = 'vit_tiny',
            teacher_model = 'vit_small',
            patch_size = 14,
            mask_probability = 0.5,
            lambda_token = 0.0,
            lambda_fea = 0.0,
            lambda_patch = 1.0,
            model_ema = True,
            model_ema_decay = 0.99996,
            model_ema_force_cpu = True,
            set_training_mode = True,
            # =====================
            architecture = 'linear',
            architecture_heads = 'proteus',
            target_temperature_list = [0.01, 0.10, 1.0],
            scale_temperature_list = False,
            sne_style = 'half',
            metric = 'MSE',
            target_mask = False,
            lambda_local = 1.0,
            lambda_global = 1.0,
            freeze_student = False,
            freeze_heads = False,
            freeze_heads_backwards = False,
            plot_norms = False,
            plot_norms_save = False,
            max_weight_decay = None,
            use_pretrained_heads = False,
            conjoin_heads = False,
            skip_loading_heads = False,
            # =====================
            **kwargs):
        super(Proteus, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        self.metrics = ['sne_global_loss', 'sne_local_loss', 'fea_loss', 'token_loss']
        self.metrics_log = [True,      True,      True,      True]
        self.metrics_save = [False,     False,     False,     False]

        student_init = self.networks.student()
        if hasattr(student_init, 'model'):
            student_init = student_init.model

        teacher_init = self.networks.teacher()

        student_model_dict = dict()
        teacher_model_dict = dict()

        embed_dim = student_init.embed_dim
        teacher_init.eval()

        student_model_dict['backbone'] = student_init
        teacher_model_dict['backbone'] = teacher_init.backbone
        
        self.embed_dim = embed_dim

        # initialize parameters and checks
        # self.total_n_global_crops = cfg.batch_size

        self.teacher = nn.ModuleDict(teacher_model_dict)
        self.student = nn.ModuleDict(student_model_dict)

        teacher_embed_dim = teacher_init.backbone.embed_dim

        self.teacher_embed_dim = teacher_embed_dim
        
        if architecture == 'linear':
            self.student_head = nn.Sequential(
                      # nn.TransformerEncoderLayer(teacher_embed_dim, 8),
                      # nn.LayerNorm(teacher_embed_dim),
                      # nn.Linear(teacher_embed_dim, embed_dim),
                      nn.Linear(embed_dim, teacher_embed_dim, bias=True))
        elif architecture == 'transformer':
            self.student_head = nn.Sequential(
                      nn.TransformerEncoderLayer(embed_dim, 8),
                      nn.LayerNorm(embed_dim),
                      nn.Linear(embed_dim, teacher_embed_dim, bias=False))
        elif architecture == 'mlp':
            self.student_head = nn.Sequential(
                      nn.Linear(embed_dim, teacher_embed_dim),
                      nn.ReLU(),
                      nn.LayerNorm(teacher_embed_dim),
                      nn.Linear(teacher_embed_dim, teacher_embed_dim),
                      nn.ReLU(),
                      nn.LayerNorm(teacher_embed_dim),
                      nn.Linear(teacher_embed_dim, teacher_embed_dim))
        elif architecture == 'identity':
            self.student_head = nn.Sequential(
                      nn.Identity())


        if hasattr(self.networks, 'teacher_head_linear') and not skip_loading_heads:
            self.teacher_head_linear = self.networks.teacher_head_linear()
        else:
            if architecture_heads == 'proteus':
                self.teacher_head_linear = nn.Sequential(
                      nn.LayerNorm(teacher_embed_dim),
                      nn.Linear(teacher_embed_dim, embed_dim, bias=True))
            elif architecture_heads == 'linear':
                self.teacher_head_linear = nn.Sequential(
                      nn.Linear(teacher_embed_dim, embed_dim, bias=False))

        if hasattr(self.networks, 'teacher_head_linear_token')  and not skip_loading_heads:
            self.teacher_head_linear_token = self.networks.teacher_head_linear_token()
        else:
            if architecture_heads == 'proteus':
                self.teacher_head_linear_token = nn.Sequential(
                      nn.LayerNorm(teacher_embed_dim),
                      nn.Linear(teacher_embed_dim, embed_dim, bias=True))
            elif architecture_heads == 'linear':
                self.teacher_head_linear_token = nn.Sequential(
                      nn.Linear(teacher_embed_dim, embed_dim, bias=False))

        if skip_loading_heads:
            self.teacher_head_linear = DummyModule(self.teacher_head_linear)
            self.teacher_head_linear_token = DummyModule(self.teacher_head_linear_token)

        if conjoin_heads:
            self.teacher_head_linear_token = self.teacher_head_linear

        self.forward_teacher = BackboneWrapper(self.teacher)
        self.forward_student = BackboneWrapper(self.student)

        if lambda_token > 0:
            self.forward_teacher_head = BackboneHeadWrapper(self.teacher, self.teacher_head_linear_token)
        else:
            self.forward_teacher_head = BackboneHeadWrapper(self.teacher, self.teacher_head_linear)

        if metric == 'Cosine':
            self.soft_criterion = CosineLoss()
        elif metric == 'MSE':
            self.soft_criterion = torch.nn.MSELoss()

        self.log_softmax = torch.nn.LogSoftmax(dim=-1)
        self.crossentropy = torch.nn.CrossEntropyLoss()

        for param in self.teacher.backbone.parameters():
            param.requires_grad = False
        if freeze_student:
            for param in self.student.backbone.parameters():
                param.requires_grad = False
        if freeze_heads:
            for param in self.teacher_head_linear.parameters():
                param.requires_grad = False
            for param in self.teacher_head_linear_token.parameters():
                param.requires_grad = False

        # if 'pretrained' in self.networks.student.keywords:
        #     if self.networks.student.keywords['pretrained'] in ['proteus_vit_ti', 'proteus_vit_s'] and use_pretrained_heads:
        #         pretrained = self.networks.student.keywords['pretrained']
        #         if pretrained == 'proteus_vit_ti':
        #             mod_state_dict = torch.load('pretrained/proteus/vit-ti.pth')
        #         elif pretrained == 'proteus_vit_s':
        #             mod_state_dict = torch.load('pretrained/proteus/vit-s.pth')
                
        #         if lambda_token > 0:
        #             mod_state_dict_use = {k: v for k, v in mod_state_dict['model'].items() if k.startswith('token_head')}
        #             new_state_dict = {}
        #             for key, value in mod_state_dict_use.items():
        #                 new_key = key.replace('token_head.', '')
        #                 new_state_dict[new_key] = value
                                
        #             self.student_head_linear_token.load_state_dict(new_state_dict)

        #         if lambda_fea > 0:
        #             mod_state_dict_use = {k: v for k, v in mod_state_dict['model'].items() if k.startswith('fea_head')}
        #             new_state_dict = {}
        #             for key, value in mod_state_dict_use.items():
        #                 new_key = key.replace('fea_head.', '')
        #                 new_state_dict[new_key] = value
                                
        #             self.student_head_linear.load_state_dict(new_state_dict)

        if scale_temperature_list:
            pb()
            target_temperature_list
            embed_dim
            teacher_embed_dim

            log_von_mises_fischer_constant_approx(1/0.01, embed_dim)
            log_von_mises_fischer_constant_approx(1/0.00137, teacher_embed_dim)

            log_von_mises_fischer_constant_approx(1/0.1, embed_dim)
            log_von_mises_fischer_constant_approx(1/0.0015, teacher_embed_dim)
            log_von_mises_fischer_constant_approx(1/0.1, teacher_embed_dim)

    # @torch.cuda.amp.autocast
    def model_step(self, batch, stage='fit'):
        with torch.cuda.amp.autocast():
            self.train(self.hparams.set_training_mode)

            if stage == 'fit':
                inputs = batch
                y = inputs['collated_global_labels']
                global_crops = inputs["collated_global_crops"]
            else:
                (global_crops,), y, idx = batch


            # ===================================================================
            # ===================================================================
            # ===================================================================
            if self.hparams.plot_norms: # Looking at attention heads
                image_size = 224
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((image_size, image_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                if False:
                    url = 'https://cdn.pixabay.com/photo/2023/08/18/15/02/dog-8198719_640.jpg'
                    import requests
                    img = Image.open(requests.get(url, stream=True).raw)
                    img = img.convert('RGB')

                    img = transform(img)

                img = torchvision.transforms.functional.to_pil_image(self.trainer.val_dataloaders.dataset.dataset.data[232]).convert("RGB")
                img = transform(img)

                attn = self.student.backbone.get_last_self_attention(img.unsqueeze(0).cuda())
                # attn = self.student.backbone.get_last_self_attention(global_crops)

                nh = attn.shape[1]
                w_featmap = 16
                h_featmap = 16
                reg_tokens = self.student.backbone.num_register_tokens
                attentions = attn[0, :, 0, (1+reg_tokens):].reshape(nh, -1)
                
                # weird: one pixel gets high attention over all heads?
                # print(torch.max(attentions, dim=1)) 
                # attentions[:, 283] = 0 
                attentions = attentions.reshape(nh, w_featmap, h_featmap)
                attentions = attentions.sum(dim=0)
                attentions = attentions/attentions.max()

                torchvision.transforms.functional.to_pil_image(attentions[:,:]).convert("RGB").show()
                # torchvision.transforms.functional.to_pil_image(global_crops[0]).convert("RGB").show()
                # pb()
            if self.hparams.plot_norms:
                image_size = 224
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((image_size, image_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                img = torchvision.transforms.functional.to_pil_image(self.trainer.val_dataloaders.dataset.dataset.data[232]).convert("RGB")
                img = transform(img)

                student_backbone_output_dict_unmask = self.student.backbone(img.unsqueeze(0).cuda(), masks=None, is_training=True)
                # student_backbone_output_dict_unmask = self.student.backbone(global_crops, masks=None, is_training=True)

                student_norms = student_backbone_output_dict_unmask['x_prenorm'][0].norm(dim=1)
                # student_norms = student_backbone_output_dict_unmask['x_norm_patchtokens'][0].norm(dim=1)
                student_norms.max()
                student_norms.min()
                reg_tokens = self.student.backbone.num_register_tokens
                print(torch.histogram(student_norms[(1+reg_tokens):].cpu(), bins=5))
                # ===========================================
                student_norms_img = student_norms[(1+reg_tokens):].reshape(16,16)
                student_norms_img = student_norms_img/student_norms_img.max()
                torchvision.transforms.functional.to_pil_image(student_norms_img).convert("RGB").show()
            if self.hparams.plot_norms:
                image_size = 224
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((image_size, image_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                img = torchvision.transforms.functional.to_pil_image(self.trainer.val_dataloaders.dataset.dataset.data[51]).convert("RGB")
                img = transform(img)

                img2 = torchvision.transforms.functional.to_pil_image(self.trainer.val_dataloaders.dataset.dataset.data[83]).convert("RGB")
                img2 = transform(img2)

                student_backbone_output_dict_unmask1 = self.student.backbone(img.unsqueeze(0).cuda(), masks=None, is_training=True)
                student_backbone_output_dict_unmask2 = self.student.backbone(img2.unsqueeze(0).cuda(), masks=None, is_training=True)
                # student_backbone_output_dict_unmask = self.student.backbone(global_crops, masks=None, is_training=True)

                student_patches1 = student_backbone_output_dict_unmask1['x_norm_patchtokens'][0]
                student_patches2 = student_backbone_output_dict_unmask2['x_norm_patchtokens'][0]
                # student_norms = student_backbone_output_dict_unmask['x_norm_patchtokens'][0].norm(dim=1)
                patch_comb = torch.cat([student_patches1, student_patches2])

                from sklearn import cluster
                clustering = cluster.KMeans()
                res = clustering.fit(patch_comb.cpu().numpy())

                clustered = res.labels_.reshape(2, w_featmap, h_featmap)
                clustered = clustered/clustered.max() * 256
                clustered = clustered.astype(np.uint8)

                import cv2

                color_mapped = cv2.applyColorMap(clustered[0], cv2.COLORMAP_JET)
                torchvision.transforms.functional.to_pil_image(color_mapped).convert("RGB").show()

                color_mapped = cv2.applyColorMap(clustered[1], cv2.COLORMAP_JET)
                torchvision.transforms.functional.to_pil_image(color_mapped).convert("RGB").show()
            if self.hparams.plot_norms:
                image_size = 224
                transform = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((image_size, image_size)),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                ])
                img = torchvision.transforms.functional.to_pil_image(self.trainer.val_dataloaders.dataset.dataset.data[51]).convert("RGB")
                img = transform(img)

                img2 = torchvision.transforms.functional.to_pil_image(self.trainer.val_dataloaders.dataset.dataset.data[83]).convert("RGB")
                img2 = transform(img2)

                student_backbone_output_dict_unmask1 = self.student.backbone(img.unsqueeze(0).cuda(), masks=None, is_training=True)
                student_backbone_output_dict_unmask2 = self.student.backbone(img2.unsqueeze(0).cuda(), masks=None, is_training=True)
                # student_backbone_output_dict_unmask = self.student.backbone(global_crops, masks=None, is_training=True)

                student_patches1 = student_backbone_output_dict_unmask1['x_norm_patchtokens'][0]
                student_patches2 = student_backbone_output_dict_unmask2['x_norm_patchtokens'][0]
                # student_norms = student_backbone_output_dict_unmask['x_norm_patchtokens'][0].norm(dim=1)
                patch_comb = torch.cat([student_patches1, student_patches2])

                patch_comb_pca = visualize_tensor_PCA(patch_comb.double(), only_calc=True, num_image=2)

                torchvision.transforms.ToPILImage()(patch_comb_pca[0]).show()
                torchvision.transforms.ToPILImage()(patch_comb_pca[1]).show()
                pb()
            # ===================================================================
            # ===================================================================
            # ===================================================================


            with torch.no_grad():
                teacher_backbone_output_dict = self.teacher.backbone(global_crops, is_training=True)
                backbone_view = teacher_backbone_output_dict["x_prenorm"]
                backbone_view = self.teacher.backbone.norm(backbone_view)

            distilled_view = self.teacher_head_linear(backbone_view)
            # ===================================================================
            # local vmf-sne
            if self.hparams.lambda_local > 0:
                distilled_D = x2p_torch_parrallel_Q_patches_calcD(distilled_view, distilled_view, mask=self.hparams.target_mask)
                backbone_D = x2p_torch_parrallel_Q_patches_calcD(backbone_view, backbone_view, mask=self.hparams.target_mask)

                tsne_loss_use = torch.tensor(0.0).to(backbone_view)
                for temp in self.hparams.target_temperature_list:

                    # we are constricting space going into a lower dimension
                    # log_von_mises_fischer_constant_approx(1/temp, self.embed_dim)
                    # log_von_mises_fischer_constant_approx(1/temp, self.teacher_embed_dim)

                    log_p12 = self.log_softmax(distilled_D/temp)
                    log_pbb12 = self.log_softmax(backbone_D/temp)

                    if self.hparams.sne_style == 'half':
                        log_P = log_pbb12
                        log_Q = log_p12

                        add_loss = torch.mean(torch.sum(torch.exp(log_P)*(log_P - log_Q), dim=(-1)))
                    elif self.hparams.sne_style == 'full':
                        log_P = torch.logsumexp(torch.stack([log_pbb12,log_pbb12.permute(0,2,1)]), dim=0)
                        log_P = log_P - torch.logsumexp(log_P, dim=(1,2), keepdims=True)
                        log_Q = torch.logsumexp(torch.stack([log_p12,log_p12.permute(0,2,1)]), dim=0)
                        log_Q = log_Q - torch.logsumexp(log_Q, dim=(1,2), keepdims=True)

                        add_loss = torch.mean(torch.sum(torch.exp(log_P)*(log_P - log_Q), dim=(1,2)))

                    tsne_loss_use += add_loss


                tsne_loss = tsne_loss_use/len(self.hparams.target_temperature_list) # record loss of just the last layer
            else:
                tsne_loss = torch.tensor(0.0)

            # =====================================
            # global vmf-sne
            backbone_view_global = backbone_view[:,0].contiguous() # class token

            distilled_view_global = self.teacher_head_linear_token(backbone_view_global)

            if self.hparams.lambda_global > 0:

                backbone_view_global_all = AllGather.apply(backbone_view_global)
                distilled_view_global_all = AllGather.apply(distilled_view_global)

                distilled_D = x2p_torch_parrallel_Q_patches_calcD(distilled_view_global, distilled_view_global_all, per_image=False, mask=self.hparams.target_mask)
                backbone_D = x2p_torch_parrallel_Q_patches_calcD(backbone_view_global, backbone_view_global_all, per_image=False, mask=self.hparams.target_mask)

                if self.hparams.sne_style == 'full':
                    distilled_D = AllGather.apply(distilled_D)
                    backbone_D = AllGather.apply(backbone_D)

                distilled_loss_use = torch.tensor(0.0).to(backbone_view)
                for temp in self.hparams.target_temperature_list:
                    log_p12 = self.log_softmax(distilled_D/temp)
                    log_pbb12 = self.log_softmax(backbone_D/temp)

                    if self.hparams.sne_style == 'half':
                        log_P = log_pbb12
                        log_Q = log_p12

                        add_loss = torch.mean(torch.sum(torch.exp(log_P)*(log_P - log_Q), dim=(-1)))
                    elif self.hparams.sne_style == 'full':
                        log_P = torch.logsumexp(torch.stack([log_pbb12,log_pbb12.T]), dim=0)
                        log_P = log_P - torch.logsumexp(log_P, dim=(0,1), keepdims=True)
                        log_Q = torch.logsumexp(torch.stack([log_p12,log_p12.T]), dim=0)
                        log_Q = log_Q - torch.logsumexp(log_Q, dim=(0,1), keepdims=True)

                        add_loss = torch.sum(torch.exp(log_P)*(log_P - log_Q), dim=(0,1))

                    distilled_loss_use += add_loss


                distillation_loss = distilled_loss_use/len(self.hparams.target_temperature_list)
            else:
                distillation_loss = torch.tensor(0.0)
            # ===================================================================
            # ===================================================================
            if self.hparams.lambda_fea > 0 or self.hparams.lambda_token > 0:
                student_output_dict = self.student.backbone(global_crops, is_training=True)
                student_view = student_output_dict["x_prenorm"]
                student_view = self.student.backbone.norm(student_view)

            if self.hparams.lambda_fea > 0:
                if self.hparams.freeze_heads_backwards:
                    fea_loss = self.soft_criterion(student_view, distilled_view.clone().detach())
                else:
                    fea_loss = self.soft_criterion(student_view, distilled_view)
            else:
                fea_loss = torch.tensor(0.0)

            if self.hparams.lambda_token > 0:
                if self.hparams.freeze_heads_backwards:
                    token_loss = self.soft_criterion(student_view[:,0,:], distilled_view_global.clone().detach())
                else:
                    token_loss = self.soft_criterion(student_view[:,0,:], distilled_view_global)
            else:
                token_loss = torch.tensor(0.0)

            # ===================================================================
            sne_local_loss = tsne_loss
            sne_global_loss = distillation_loss
            # ====================================================
            total_loss_project = self.hparams.lambda_local*sne_local_loss + self.hparams.lambda_global* sne_global_loss \
                + self.hparams.lambda_fea * fea_loss + self.hparams.lambda_token * token_loss

        # if torch.distributed.get_rank() == 0:
        #     pb()

        total_loss = total_loss_project
        # pb()
        results_dict = {
            'loss': total_loss,
            'sne_local_loss': sne_local_loss, 'sne_global_loss': sne_global_loss, 'fea_loss': fea_loss , 'token_loss': token_loss
        }
        return results_dict

    # =====================================================
    # =====================================================

    def on_predict_epoch_start(self):
        if self.hparams.plot_norms_save:
            super(Proteus, self).on_predict_epoch_start()
            # iterate over batches and obtain prototypes
            datamodule = self.trainer.datamodule
            transform = datamodule.aug_predict

            dataloader_kNN = datamodule._predict_dataloader()
            dataset = copy.deepcopy(datamodule.data_val)

            dataset = LightlyDataset.from_torch_dataset(dataset, transform, datamodule.aug_targets)
            self.ref_dataset = dataset
            dataloader_kNN.keywords['shuffle'] = False # obselete
            dataloader_kNN.keywords['drop_last'] = False # obselete
            dataloader_kNN = datamodule.predict_dataloader(dataset=dataset, base_dataloader=dataloader_kNN)

            train_data = dataloader_kNN

            if dist.is_initialized() and dist.get_world_size() > 1:
                rank = dist.get_rank()
            else:
                rank = 0
            miniters = self.trainer.log_every_n_steps
            if isinstance(self.trainer.limit_val_batches, int):
                self.limit_batches = True
            else:
                self.limit_batches = False

            self.dp = torch.tensor(0.0)
            self.dp = self.dp.to(next(self.student.backbone.parameters()).device)

            train_feature_norms = []
            with torch.no_grad():
                for i,data in enumerate(tqdm.tqdm(train_data, position=rank, miniters=miniters)):
                    [img], target, idx = data
                    img, target = img.to(self.dp.device), target.to(self.dp.device)

                    student_backbone_output_dict_unmask = self.student.backbone(img, masks=None, is_training=True)

                    student_norms = student_backbone_output_dict_unmask['x_prenorm'].norm(dim=-1)

                    reg_tokens = self.student.backbone.num_register_tokens
                    student_norms = student_norms[:, (1+reg_tokens):]
                    student_norms = student_norms.reshape(-1)
                    # print(torch.histogram(student_norms.cpu(), bins=5))

                    train_feature_norms.append(student_norms)
                    if self.limit_batches:
                        if i == self.trainer.limit_val_batches:
                            break

            logger = self.trainer.logger
            output_folder = os.path.join(logger.save_dir, logger.name, 'version_'+str(logger.version), 'predictions')
            Path(output_folder).mkdir(parents=True, exist_ok=True)
            output_file_base = 'epoch='+str(self.trainer.current_epoch)+'-step='+str(self.trainer.global_step)+'-'

            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
            else:
                rank = 0
            
            output_file_final = output_folder + '/' + output_file_base + '_predictions'+str(rank) +'_'

            train_feature_norms = torch.cat(train_feature_norms, dim=0).contiguous()
            train_feature_norms.cpu().numpy().dump(output_file_final + 'student_norms.npy')
            sys.exit()

    # ==========================================================
    # ==========================================================
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        # if self.model_ema is not None:
        #     self.model_ema.update(self.student.backbone)

        if self.hparams.max_weight_decay is not None:
            for group in self.trainer.optimizers[0].param_groups:
                if group['weight_decay'] > 0:
                    if 'weight_decay_orig' not in group:
                        group['weight_decay_orig'] = group['weight_decay']
                    group['weight_decay'] = compute_cosine(group['weight_decay_orig'], self.hparams.max_weight_decay, self.trainer.current_epoch, self.trainer.max_epochs)
                    # pb()

    # ==========================================================
    # ==========================================================

    def predict_step(self, batch: Any, batch_idx: int):
        x = batch[0][0]
        idx = batch[2]
        res = self.forward_student(x)
        return res, idx

    # ==========================================================
    # ==========================================================

def compute_cosine(base_value, final_value, step, max_steps):
    if base_value < final_value:
        sign = -1
    else:
        sign = 1

    return final_value + sign*0.5 * np.abs(base_value - final_value) * \
        (1 + np.cos(np.pi * (step) / (max_steps)))
