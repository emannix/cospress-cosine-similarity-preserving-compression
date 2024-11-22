from pdb import set_trace as pb
import torch
import numpy as np
import pandas as pd
from torchvision.datasets.folder import default_loader

from lightning import LightningDataModule
from .base_lightning_module import BaseDataModule
import os
from PIL import Image
from torchvision import tv_tensors

from . import helper

class CSVDataset(torch.utils.data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, image_folder='test/',
        metadata_csv_file='test.csv', metadata_csv_image = 'image_id', 
        metadata_csv_label = 'class_id', image_ext = '.png', 
        embedding_file = None, segmentation = None,
        transform=None, target_transform=None, pre_target_transform=None):
        'Initialization'
        
        if metadata_csv_file is not None:
            if type(metadata_csv_file) is list:
                data_sheet_list = []
                for csv in metadata_csv_file:
                    data_sheet_list.append(pd.read_csv(csv, encoding='utf-8', keep_default_na=False))
                data_sheet = pd.concat(data_sheet_list)
            else:
                data_sheet = pd.read_csv(metadata_csv_file, encoding='utf-8', keep_default_na=False)

            if isinstance(metadata_csv_image, list):
              cols = data_sheet[metadata_csv_image].agg(''.join, axis=1)+image_ext
            else:
              cols = data_sheet[metadata_csv_image] +image_ext
            imgs = [os.path.join(image_folder, x) for x in cols]
        else:
            imgs = os.listdir(image_folder)
            imgs = [x for x in imgs if image_ext in x]

        self.segmentation = segmentation
        if segmentation is None:
            if metadata_csv_label is None:
                labels = np.zeros(len(imgs))
            else:
                labels = data_sheet[metadata_csv_label].values
            if pre_target_transform is not None:
                labels = pre_target_transform(labels)

            # if labels.dtype.char == 'O':
            #     # labels = labels.astype('U')
            self.target_labels = np.unique(labels) # return_counts=True
            self.target_labels.sort()
            # labels = np.argwhere(labels[:, None] == self.target_labels[None, :])[:,1]
            labels_array = labels[:, None] == self.target_labels[None, :]
            labels = np.argmax(labels_array, axis=1).astype(int)
            missing_labels = labels_array.sum(axis=1)
            labels[missing_labels == 0] = -1

            # ===========================================
            print(self.target_labels)
            print(np.unique(labels, return_counts=True))
            # ===========================================

            self.targets = torch.tensor(labels)
            self.pre_target_transform = pre_target_transform
        else:
            if isinstance(metadata_csv_label, list):
              cols = data_sheet[metadata_csv_label].agg(''.join, axis=1)+image_ext
            else:
              cols = data_sheet[metadata_csv_label] +image_ext
            targets = [os.path.join(image_folder, x)  if x != '' else '' for x in cols]
            self.targets = targets
            # for i in range(len(self.targets)):
            #     if self.targets[i] != '':
            #         image = Image.open(self.targets[i])
            #         if image.mode == 'L':
            #             num_classes = 2
            #         elif image.mode == 'P':
            #             palette = image.getpalette()
            #             pb()
            #             num_classes = len(palette)//3
            num_classes = self.segmentation['num_classes']
            self.target_labels = torch.arange(num_classes) # self.segmentation['num_classes']

        self.inputs = np.array(imgs)
        self.transform = transform
        self.target_transform = target_transform

        if embedding_file is not None:
            self.embedding = np.load(embedding_file)
 
  def __len__(self):
        'Denotes the total number of samples'
        return len(self.inputs)

  def __getitem__(self, index):
        'Generates one sample of data'
        target = self.targets[index]
        if self.target_transform is not None:
            target = self.target_transform(target)
        img_path = self.inputs[index]
        image = default_loader(img_path)
        if self.segmentation is None:
            if self.transform is not None:
                image = self.transform(image)
        else:
            if target == '':
                if self.target_labels.shape[0] == 2:
                    target = Image.new('L', image.size)
                elif self.target_labels.shape[0] > 2:
                    target = Image.new('P', image.size)
            else:
                target = Image.open(target)
            image = tv_tensors.Image(image)
            target = tv_tensors.Mask(target)
            if self.transform is not None:
                res = self.transform([image, target])
                if (len(res) == 2):
                    image = res[0]
                    target = res[1]
                    if len(target) == 1:
                        target = target[0].squeeze()
                else:
                    return res
            if self.target_labels.shape[0] == 2:
                target[target == 255] = 1
        return image, target

# ========================================================
