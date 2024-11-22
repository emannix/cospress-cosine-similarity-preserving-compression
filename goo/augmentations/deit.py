
from lightly.transforms.multi_view_transform import MultiViewTransform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform
from torchvision import transforms

from pdb import set_trace as pb

class BuildTransform(MultiViewTransform):
    def __init__(self, is_train, input_size = 224, color_jitter=0.3, 
            aa='rand-m9-mstd0.5-inc1',
            train_interpolation='bicubic', reprob=0.25, remode='pixel', recount=1,
            eval_crop_ratio=0.875, scale=None):
        resize_im = input_size > 32
        if is_train:
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=input_size,
                scale=scale,
                is_training=True,
                color_jitter=color_jitter,
                auto_augment=aa,
                interpolation=train_interpolation,
                re_prob=reprob,
                re_mode=remode,
                re_count=recount,
            )
            if not resize_im:
                # replace RandomResizedCropAndInterpolation with
                # RandomCrop
                transform.transforms[0] = transforms.RandomCrop(
                    input_size, padding=4)
        else:
            t = []
            if resize_im:
                size = int(input_size / eval_crop_ratio)
                t.append(
                    transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
                )
                t.append(transforms.CenterCrop(input_size))

            t.append(transforms.ToTensor())
            t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
            transform = transforms.Compose(t)
        super().__init__(transforms=[transform])

    # def __call__(self, image):
    #     t = [transform(image) for transform in self.transforms]
    #     pb()
    #     return t