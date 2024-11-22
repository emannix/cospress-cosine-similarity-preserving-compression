
from pdb import set_trace as pb
import torch
import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader

from lightning import LightningDataModule
from .base_lightning_module import BaseDataModule
import os

from . import helper
from .base_csv_dataset_image import CSVDataset
from .base_csv_dataset_embedded import CSVDatasetEmbedding

class CSVDataModule(BaseDataModule):
    def __init__(
        self,
        base_dir = "./",
        data_dir = "data/",
        train_folder = 'training/',
        train_csv_file='test.csv', train_csv_image = 'image_id', 
        train_csv_label = 'class_id', train_image_ext = '.png',
        train_embedding_file = None,
        val_folder = 'testing/',
        val_csv_file='test.csv', val_csv_image = 'image_id', 
        val_csv_label = 'class_id', val_image_ext = '.png',
        val_embedding_file = None,
        test_folder = None,
        test_csv_file=None, test_csv_image = None, 
        test_csv_label = None, test_image_ext = None,
        test_embedding_file = None,
        target_transform=None, pre_target_transform=None,
        media='image',
        segmentation = None,
        **kwargs):
        super().__init__(**kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)


    def setup(self, stage= None):
        
        self.hparams.base_dir = helper.set_base_dir(self.hparams.base_dir)
        base_dir = self.hparams.base_dir + self.hparams.data_dir
        train_folder = helper.combine_directories(base_dir, self.hparams.train_folder)
        train_csv_file = helper.combine_directories(base_dir, self.hparams.train_csv_file)
        if self.hparams.train_embedding_file is not None:
            train_embedding_file = helper.combine_directories(base_dir, self.hparams.train_embedding_file)
        else:
            train_embedding_file = None
        val_folder = helper.combine_directories(base_dir, self.hparams.val_folder)
        val_csv_file = helper.combine_directories(base_dir, self.hparams.val_csv_file)
        if self.hparams.val_embedding_file is not None:
            val_embedding_file = helper.combine_directories(base_dir, self.hparams.val_embedding_file)
        else:
            val_embedding_file = None

        if self.hparams.test_csv_file is not None:
            test_csv_file = helper.combine_directories(base_dir, self.hparams.test_csv_file)
            test_folder = helper.combine_directories(base_dir, self.hparams.test_folder)
        else:
            test_csv_file = None
            test_folder = None
        if self.hparams.test_embedding_file is not None:
            test_embedding_file = helper.combine_directories(base_dir, self.hparams.test_embedding_file)
        else:
            test_embedding_file = None

        if self.hparams.media == 'image':
            loader = CSVDataset
        elif self.hparams.media == 'embedding':
            loader = CSVDatasetEmbedding

        self.data_train = loader(
            image_folder = train_folder,
            metadata_csv_file = train_csv_file, 
            metadata_csv_image = self.hparams.train_csv_image, 
            metadata_csv_label = self.hparams.train_csv_label, image_ext = self.hparams.train_image_ext,
            target_transform = self.hparams.target_transform, pre_target_transform=self.hparams.pre_target_transform,
            embedding_file = train_embedding_file, segmentation=self.hparams.segmentation
            )

        self.data_val = loader(
            image_folder = val_folder,
            metadata_csv_file = val_csv_file, 
            metadata_csv_image = self.hparams.val_csv_image, 
            metadata_csv_label = self.hparams.val_csv_label, image_ext = self.hparams.val_image_ext,
            target_transform = self.hparams.target_transform, pre_target_transform=self.hparams.pre_target_transform,
            embedding_file = val_embedding_file, segmentation=self.hparams.segmentation
            )

        if test_csv_file is not None:
            self.data_test = loader(
                image_folder = test_folder,
                metadata_csv_file = test_csv_file, 
                metadata_csv_image = self.hparams.test_csv_image, 
                metadata_csv_label = self.hparams.test_csv_label, image_ext = self.hparams.test_image_ext,
                target_transform = self.hparams.target_transform, pre_target_transform=self.hparams.pre_target_transform,
                embedding_file = test_embedding_file, segmentation=self.hparams.segmentation
                )
        else:
            self.data_test = self.data_val

        # assume training set is one that makes truth
        val_labels_synced = self.data_train.target_labels == self.data_val.target_labels
        test_labels_synced = self.data_train.target_labels == self.data_test.target_labels
        if not isinstance(val_labels_synced, bool):
            val_labels_synced = all(val_labels_synced)
        if not isinstance(test_labels_synced, bool):
            test_labels_synced = all(test_labels_synced)
            
        if not val_labels_synced or not test_labels_synced:
            labels = self.data_train.target_labels
            index_to_class = {label: index  for index, label in enumerate(labels)}
            print('syncing labels')

            test_labels = self.data_test.target_labels[self.data_test.targets]
            test_indices = np.array([index_to_class[key] for key in test_labels])
            self.data_test.target_labels = self.data_train.target_labels
            self.data_test.targets = test_indices

            val_labels = self.data_val.target_labels[self.data_val.targets]
            val_indices = np.array([index_to_class[key] for key in val_labels])
            self.data_val.target_labels = self.data_train.target_labels
            self.data_val.targets = val_indices


        # =======================================================
        super().setup(stage)


