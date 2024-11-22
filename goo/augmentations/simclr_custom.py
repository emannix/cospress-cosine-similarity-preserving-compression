# https://github.com/lightly-ai/lightly/blob/master/lightly/data/collate.py

from lightly.transforms.rotation import random_rotation_transform
from lightly.transforms import GaussianBlur as GaussianBlurLightly
from lightly.transforms.utils import IMAGENET_NORMALIZE
from lightly.transforms.multi_view_transform import MultiViewTransform

imagenet_normalize = IMAGENET_NORMALIZE

from .simclr_helpers import Clip, GaussianBlur

from torchvision import transforms
from pdb import set_trace as pb
import torchvision.transforms as T
from typing import Optional, Tuple, Union

import numpy as np
import torch
import random

class SimCLRCustom(MultiViewTransform):
    def __init__(
        self,
        copies=2,
        input_size: int = 224,
        cj_prob: float = 0.8,
        cj_strength: float = 0.5,
        min_scale: float = 0.08,
        random_gray_scale: float = 0.2,
        gaussian_blur: float = 0.5,
        guassian_blur_lightly_version: bool = True,
        kernel_size: Optional[float] = None,
        sigmas: Tuple[float, float] = (0.2, 2),
        vf_prob: float = 0.0,
        hf_prob: float = 0.5,
        rr_prob: float = 0.0,
        rr_degrees: Union[None, float, Tuple[float, float]] = None,
        normalize: dict = imagenet_normalize,
        static_seed = None,
    ):
        self.static_seed = static_seed
        if isinstance(input_size, tuple):
            input_size_ = max(input_size)
        else:
            input_size_ = input_size

        cj_bright, cj_contrast, cj_sat = 0.8*cj_strength, 0.8*cj_strength, 0.8*cj_strength
        cj_hue = 0.2*cj_strength

        color_jitter = T.ColorJitter(cj_bright, cj_contrast, cj_sat, cj_hue)

        transform = [
            T.RandomResizedCrop(size=input_size, scale=(min_scale, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            random_rotation_transform(rr_prob=rr_prob, rr_degrees=rr_degrees),
            T.RandomHorizontalFlip(p=hf_prob),
            T.RandomVerticalFlip(p=vf_prob),
            T.RandomApply([color_jitter], p=cj_prob),
            T.RandomGrayscale(p=random_gray_scale),
        ]

        if guassian_blur_lightly_version:
            gb = GaussianBlurLightly(kernel_size=kernel_size, sigmas=sigmas, prob=gaussian_blur)
            transform += [gb, T.ToTensor()]
        else:
            gb = GaussianBlur(input_size // 10, gaussian_blur)
            transform += [T.ToTensor(), gb]

        if normalize:
            transform += [T.Normalize(mean=normalize["mean"], std=normalize["std"])]

        transform = T.Compose(transform)
        super().__init__(transforms=[transform for x in range(copies)])

    def __call__(self, image):
        if self.static_seed is not None:
            torch.manual_seed(self.static_seed)
            np.random.seed(self.static_seed)
            random.seed(self.static_seed)
        t = [transform(image) for transform in self.transforms]
        return t