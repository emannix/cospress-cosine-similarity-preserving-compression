
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
import torchvision
from PIL import Image
import numpy as np

import copy
from ...lightly.dataset import LightlyDataset
import tqdm
import os
from pathlib import Path
import sys

from timm.models.deit import VisionTransformerDistilled

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

class BackboneWrapper(nn.Module):
    def __init__(self, teacher):
        super(BackboneWrapper, self).__init__()
        self.teacher = teacher

    def forward(self, x):
        if isinstance(self.teacher.backbone, VisionTransformerDistilled):
            with torch.no_grad():
                teacher_backbone_output_dict = self.teacher.backbone(x)
            backbone_cls_view = teacher_backbone_output_dict
        else:
            with torch.no_grad():
                    teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)

            backbone_cls_view = teacher_backbone_output_dict["x_norm_clstoken"]
            
        # backbone_cls_view = torch.nn.functional.normalize(backbone_cls_view, dim=1)
        # backbone_cls_view @ backbone_cls_view.T
        # (backbone_cls_view @ backbone_cls_view.T).min()
        # -0.1757
        # (backbone_cls_view @ backbone_cls_view.T).mean()
        # 0.2580
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
            lambda_token = 1.0,
            lambda_fea = 1.0,
            lambda_patch = 1.0,
            model_ema = True,
            model_ema_decay = 0.99996,
            model_ema_force_cpu = True,
            set_training_mode = True,
            # ======================
            metric='MSE',
            max_weight_decay=None,
            use_pretrained_heads=True,
            freeze_student = False,
            skip_loading_heads = False,
            # ======================
            plot_norms = False,
            plot_norms_save = False,
            evaluate_val = True,
            **kwargs):
        super(Proteus, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        self.metrics = ['patch_loss', 'fea_loss', 'token_loss']
        self.metrics_log = [True,      True,      True]
        self.metrics_save = [False,     False,     False]

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

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        teacher_embed_dim = teacher_init.backbone.embed_dim

        if hasattr(self.networks, 'patch_head') and not skip_loading_heads:
            self.patch_head = self.networks.patch_head()
        else:
            self.patch_head = nn.Sequential(
                  nn.LayerNorm(embed_dim),
                  nn.Linear(embed_dim, teacher_embed_dim))
        
        if hasattr(self.networks, 'token_head') and not skip_loading_heads:
            self.token_head = self.networks.token_head()
        else:
            self.token_head = nn.Sequential(
                  nn.LayerNorm(embed_dim),
                  nn.Linear(embed_dim, teacher_embed_dim))

        if hasattr(self.networks, 'fea_head') and not skip_loading_heads:
            self.fea_head = self.networks.fea_head()
        else:
            self.fea_head = nn.Sequential(
                      nn.LayerNorm(embed_dim),
                      nn.Linear(embed_dim, teacher_embed_dim))

        if skip_loading_heads:
            self.patch_head = DummyModule(self.patch_head)
            self.token_head = DummyModule(self.token_head)
            self.fea_head = DummyModule(self.fea_head)

        if metric == 'Cosine':
            self.soft_criterion = CosineLoss()
        elif metric == 'MSE':
            self.soft_criterion = torch.nn.MSELoss()

        for param in self.teacher.backbone.parameters():
            param.requires_grad = False
        if freeze_student:
            for param in self.student.backbone.parameters():
                param.requires_grad = False

        self.model_ema = None
        if model_ema:
            # Important to create EMA model after cuda(), DP wrapper, and AMP but before SyncBN and DDP wrapper
            self.model_ema = ModelEma(
                self.student.backbone,
                decay=model_ema_decay,
                device='cpu' if model_ema_force_cpu else '',
                resume='')

        if 'pretrained' in self.networks.student.keywords:
            if self.networks.student.keywords['pretrained'] in ['proteus_vit_ti', 'proteus_vit_s'] and use_pretrained_heads:
                pretrained = self.networks.student.keywords['pretrained']
                if pretrained == 'proteus_vit_ti':
                    mod_state_dict = torch.load('pretrained/proteus/vit-ti.pth')
                elif pretrained == 'proteus_vit_s':
                    mod_state_dict = torch.load('pretrained/proteus/vit-s.pth')
                            
                mod_state_dict_use = {k: v for k, v in mod_state_dict['model'].items() if k.startswith('token_head')}
                new_state_dict = {}
                for key, value in mod_state_dict_use.items():
                    new_key = key.replace('token_head.', '')
                    new_state_dict[new_key] = value
                            
                self.token_head.load_state_dict(new_state_dict)

                mod_state_dict_use = {k: v for k, v in mod_state_dict['model'].items() if k.startswith('fea_head')}
                new_state_dict = {}
                for key, value in mod_state_dict_use.items():
                    new_key = key.replace('fea_head.', '')
                    new_state_dict[new_key] = value
                            
                self.fea_head.load_state_dict(new_state_dict)

                mod_state_dict_use = {k: v for k, v in mod_state_dict['model'].items() if k.startswith('ibot_head')}
                new_state_dict = {}
                for key, value in mod_state_dict_use.items():
                    new_key = key.replace('ibot_head.', '')
                    new_state_dict[new_key] = value
                            
                self.patch_head.load_state_dict(new_state_dict)

        self.forward_student = BackboneWrapper(self.student)
        self.forward_student_head = BackboneHeadWrapper(self.student, self.token_head)

        
    # @torch.cuda.amp.autocast
    def model_step(self, batch, stage='fit'):
        with torch.cuda.amp.autocast():
            self.train(self.hparams.set_training_mode)

            if stage == 'fit' or self.hparams.evaluate_val:
                inputs = batch
                y = inputs['collated_global_labels']
                global_crops = inputs["collated_global_crops"]

                masks = inputs["collated_masks"]
                mask_indices_list = inputs["mask_indices_list"]
                n_masked_patches = mask_indices_list.shape[0]
                upperbound = inputs["upperbound"]

                n_global_crops = 1

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

                # compute teacher output
                # @torch.no_grad()
                def compute_teacher_output():
                    with torch.no_grad():
                        teacher_backbone_output_dict = self.teacher.backbone(global_crops, is_training=True)
                    teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
                    teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]

                    _dim = teacher_patch_tokens.shape[-1]

                    # mask teacher patch tokens
                    buffer_tensor_teacher = teacher_patch_tokens.new_zeros(upperbound, _dim)
                    torch.index_select(
                        teacher_patch_tokens.flatten(0, 1),
                        dim=0,
                        index=mask_indices_list,
                        out=buffer_tensor_teacher[:n_masked_patches],
                    )
                    teacher_patch_tokens_masked = buffer_tensor_teacher[:n_masked_patches]

                    return teacher_cls_tokens, teacher_patch_tokens, teacher_patch_tokens_masked

                # get the teacher outputs
                (
                    teacher_cls_tokens,
                    teacher_patch_tokens,
                    teacher_patch_tokens_masked
                ) = compute_teacher_output()
                
                cur_masks = masks if self.hparams.mask_probability > 0 else None

                student_backbone_output_dict, student_backbone_output_dict_unmask = self.student.backbone(
                    [global_crops, global_crops], masks=[cur_masks, None], is_training=True
                )

                student_cls_token_unmask = student_backbone_output_dict_unmask["x_norm_clstoken"]
                student_patch_tokens_unmask = student_backbone_output_dict_unmask["x_norm_patchtokens"]
                student_patch_tokens = student_backbone_output_dict["x_norm_patchtokens"]

                if False: # Looking at PCA maps and Norms
                    # PCA Maps
                    visualize_tensor_PCA(teacher_patch_tokens[0].double()).show()
                    visualize_tensor_PCA(student_patch_tokens_unmask[0].double()).show()
                    student_cls_token_unmask_proj = self.fea_head(student_patch_tokens_unmask)
                    visualize_tensor_PCA(student_cls_token_unmask_proj[0].double()).show()
                    # Image.open(self.trainer.val_dataloaders.dataset.dataset._images[0]).convert("RGB").show()
                    torchvision.transforms.functional.to_pil_image(self.trainer.val_dataloaders.dataset.dataset.data[0]).convert("RGB").show()
                    # Norms
                    teacher_patch_tokens[0].norm(dim=1).max()
                    teacher_patch_tokens[0].norm(dim=1).min()

                    student_patch_tokens_unmask[0].norm(dim=1).max()
                    student_patch_tokens_unmask[0].norm(dim=1).min()
                    # There doesn't seem to be any outliers in the norms...

                    with torch.no_grad():
                        teacher_backbone_output_dict = self.teacher.backbone(global_crops, is_training=True)
                    teacher_backbone_output_dict['x_prenorm'][0].norm(dim=1).max()
                    teacher_backbone_output_dict['x_prenorm'][0].norm(dim=1).min()

                    student_backbone_output_dict_unmask['x_prenorm'][0].norm(dim=1).max()
                    student_backbone_output_dict_unmask['x_prenorm'][0].norm(dim=1).min()
                    # Does seem to be outliers in these norms
                    torch.histogram(student_backbone_output_dict_unmask['x_prenorm'][0].norm(dim=1).cpu(), bins=5)
                    # ===========================================

                # mask student patch tokens
                _dim = student_patch_tokens.shape[-1]
                
                buffer_tensor_student = student_patch_tokens.new_zeros(upperbound, _dim)
                buffer_tensor_student[:n_masked_patches].copy_(
                    torch.index_select(student_patch_tokens.flatten(0, 1),
                                        dim=0,
                                        index=mask_indices_list)
                )

                ## projection head
                student_patch_tokens_unmask = self.fea_head(student_patch_tokens_unmask)
                
                student_cls_token_unmask = self.token_head(student_cls_token_unmask)
                
                tokens_after_head = self.patch_head(buffer_tensor_student)
                student_patch_tokens_masked = tokens_after_head[:n_masked_patches]

                ## token objective
                distillation_loss_token = self.soft_criterion(student_cls_token_unmask, teacher_cls_tokens)

                ## fea objective
                student_whole_fea = torch.cat((student_cls_token_unmask.unsqueeze(1),student_patch_tokens_unmask),dim=1)
                teacher_whole_fea = torch.cat((teacher_cls_tokens.unsqueeze(1),teacher_patch_tokens),dim=1)
                distillation_loss_fea = self.soft_criterion(student_whole_fea, teacher_whole_fea)

                ## patch objective
                patch_loss = self.soft_criterion(student_patch_tokens_masked, teacher_patch_tokens_masked)
                
                # coefficient
                token_loss = self.hparams.lambda_token * distillation_loss_token
                fea_loss = self.hparams.lambda_fea * distillation_loss_fea
                patch_loss = self.hparams.lambda_patch * patch_loss

                # compute the total loss
                total_loss = patch_loss + fea_loss + token_loss
            else:
                total_loss = torch.tensor(0.0)
                patch_loss = torch.tensor(0.0)
                fea_loss = torch.tensor(0.0)
                token_loss = torch.tensor(0.0)

        # pb()
        results_dict = {
            'loss': total_loss,
            'patch_loss': patch_loss, 'fea_loss': fea_loss, 'token_loss': token_loss
        }
        return results_dict

    # ==========================================================
    # ==========================================================
    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.model_ema is not None:
            self.model_ema.update(self.student.backbone)

        if self.hparams.max_weight_decay is not None:
            for group in self.trainer.optimizers[0].param_groups:
                if group['weight_decay'] > 0:
                    if 'weight_decay_orig' not in group:
                        group['weight_decay_orig'] = group['weight_decay']
                    group['weight_decay'] = compute_cosine(group['weight_decay_orig'], self.hparams.max_weight_decay, self.trainer.current_epoch, self.trainer.max_epochs)
                    # pb()

    # ==========================================================
    # ==========================================================

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