
from torchvision.datasets import Places365
from pdb import set_trace as pb
import torch

from torchvision.datasets import ImageFolder

def BaseLoader(root, min_index=0, max_index=-1, **kwargs):
    # data_train = Places365(root=root, split='train-standard', download=True)
    # data_val = Places365(root=root, split='val', download=True)

    data_train = ImageFolder(root=root)

    data_train.samples = data_train.samples
    data_train.samples = data_train.samples[min_index:max_index]

    data_train.targets = [data_train.samples[i][1] for i in range(len(data_train.samples))]
    data_train.imgs = data_train.samples

    return data_train
