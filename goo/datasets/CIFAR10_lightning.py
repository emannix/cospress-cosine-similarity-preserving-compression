
from lightning import LightningDataModule
from .base_lightning_module import BaseDataModule

from torchvision import datasets

class CIFAR10DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir = "data/", **kwargs):
        super().__init__(**kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    @property
    def num_classes(self):
        return 10

    def setup(self, stage = None):
        
        self.data_train = datasets.CIFAR10(root=self.hparams.data_dir, train=True, download=True)
        self.data_val = datasets.CIFAR10(root=self.hparams.data_dir, train=False, download=True)
        self.data_test = self.data_val

        # =======================================================
        super().setup(stage)


