# https://github.com/emannix/Proteus-pytorch/tree/main/pretrain

import os
import json
from torchvision.datasets import DatasetFolder
from torchvision.io import read_image
from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from PIL import Image

import numpy as np

import random
import numpy as np
from torchvision import transforms
import torch
import torchvision

from abc import ABC

from pdb import set_trace as pb

class SetSeedTransform(torch.nn.Module):
    def forward(self, img): 
        seed = 0
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        return img

def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        if args.augmentations_fix_seed:
            transform.transforms.insert(0, SetSeedTransform())
        # noise_3d = torchvision.transforms.functional.to_pil_image(torch.randn(3, 224, 224))
        # transform(noise_3d)
        return transform

    if args.augmentations_fix_seed:
        t = [SetSeedTransform()]
    else:
        t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    t = transforms.Compose(t)
    # noise_3d = torch.randn(3, 224, 224)
    # pb()
    return t

def collate_data_and_cast_aug(
    samples_list,
    mask_ratio,
    mask_probability,
    dtype,
    n_tokens=None,
    mask_first_n=False,
    mask_generator=None,
    clone_batch=1,
):
    if dtype == 'torch.half':
        dtype = torch.half
    elif dtype == 'torch.float':
        dtype = torch.float

    n_global_crops = 1

    assert n_global_crops > 0, "global crops number should be > 0"
    collated_global_crops = torch.stack([s[i] for i in range(n_global_crops) for s in samples_list])

    labels = [s[1] for s in samples_list]
    labels = torch.LongTensor(labels)
    collated_global_labels = labels.repeat(n_global_crops)

    B = len(collated_global_crops)
    N = n_tokens
    n_samples_masked = int(B * mask_probability)

    masks_list = []
    upperbound = 0

    masks_enc = torch.full((1,), 0, dtype=torch.int32)
    masks_pred = torch.full((1,), 0, dtype=torch.int32)
    # specify the number of masks to append
    number_masks = n_samples_masked * clone_batch
    # do per-sample masking
    if isinstance(mask_ratio, (tuple, list)) and len(mask_ratio) == 2:
        probs = torch.linspace(*mask_ratio, number_masks + 1)
        for i in range(0, number_masks):
            prob_min = probs[i]
            prob_max = probs[i + 1]
            masks_list.append(torch.BoolTensor(mask_generator(int(N * random.uniform(prob_min, prob_max)))))
            upperbound += int(N * prob_max)
    else:
        mask_ratio = mask_ratio[0]
        # apply the same mask ratio to all images
        for i in range(0, number_masks):
            masks_list.append(torch.BoolTensor(mask_generator(int(N * mask_ratio))))
            upperbound += int(N * mask_ratio)

    # append masks for unmasked samples
    for i in range(n_samples_masked, B):
        # masks_list.append(torch.BoolTensor(mask_generator(0)))
        masks_list.append(torch.BoolTensor(mask_generator.get_none_mask()))

    if not mask_first_n and mask_probability > 0.0:  # shuffle masking -- not shuffling for mae-style
        random.shuffle(masks_list)

    collated_masks = torch.stack(masks_list).flatten(1)
    mask_indices_list = collated_masks.flatten().nonzero().flatten()

    masks_weight = (1 / collated_masks.sum(-1).clamp(min=1.0)).unsqueeze(-1).expand_as(collated_masks)[collated_masks]

    return {
        "collated_global_crops": collated_global_crops.to(dtype),
        "collated_global_labels": collated_global_labels,
        "collated_masks": collated_masks,
        "mask_indices_list": mask_indices_list,
        "masks_weight": masks_weight,
        "upperbound": upperbound,
        "n_masked_patches": torch.full((1,), fill_value=mask_indices_list.shape[0], dtype=torch.long),
        "masks_enc": masks_enc,
        "masks_pred": masks_pred,
    }




class MaskingGenerator(ABC):
    def __init__(self, input_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width

    def __repr__(self):
        raise NotImplementedError

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        raise NotImplementedError

    def get_none_mask(self):
        return np.zeros(shape=self.get_shape(), dtype=bool)
    
    
    
class RandomMaskingGenerator(MaskingGenerator):
    def __init__(
        self,
        input_size,
    ):
        """
        Args:
            input_size: the size of the token map, e.g., 14x14
        """
        super().__init__(input_size)

    def __repr__(self):
        repr_str = f"Random Generator({self.height}, {self.width})"
        return repr_str

    def _mask(self, mask, max_mask_patches):
        return super()._mask(mask, max_mask_patches)

    def __call__(self, num_masking_patches=0):
        if num_masking_patches <= 0:
            return np.zeros(shape=self.get_shape(), dtype=bool)

        mask = np.hstack([np.ones(num_masking_patches, dtype=bool),
                          np.zeros(self.num_patches - num_masking_patches, dtype=bool)])
        np.random.shuffle(mask)
        mask = mask.reshape(self.get_shape())
        return mask



