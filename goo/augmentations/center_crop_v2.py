
from torchvision.transforms import v2
from pdb import set_trace as pb

from ..lightly.multi_view_transform_v2 import MultiViewTransform
from lightly.transforms.utils import IMAGENET_NORMALIZE
from torchvision import tv_tensors
from .torchvision_v2_helpers import ConditionalTransform, toTensorv2

imagenet_normalize = IMAGENET_NORMALIZE

class CenterCropAugment(MultiViewTransform):
    def __init__(
        self,
        crop_size = 224, resize=256, stretch=False, normalize=imagenet_normalize, views=1, **kwargs
    ):
        if stretch:
            resize_use = (resize, resize)
        else:
            resize_use = resize

        transform =  [
                v2.Resize(size=resize_use,
                    interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
                v2.CenterCrop(size=crop_size) if crop_size is not None else None,
                ConditionalTransform(toTensorv2(), tv_tensors.Image)
            ]

        if normalize:
            transform += [ConditionalTransform(v2.Normalize(mean=normalize["mean"], std=normalize["std"]), tv_tensors.Image)]

        transform = v2.Compose([x for x in transform if x is not None])

        transform_list = [transform]
        copies_list = [views]

        super().__init__(transforms=transform_list, copies=copies_list)
