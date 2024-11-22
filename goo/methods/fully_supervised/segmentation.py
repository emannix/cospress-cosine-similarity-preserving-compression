
from ..model_base_s import ModelBaseS
from pdb import set_trace as pb

from ...networks.dinov2_linear_classifier import create_linear_input

from torchmetrics.functional.classification import multiclass_accuracy
from torch.nn.functional import softmax
import torch.nn as nn
import torch

import numpy as np
import matplotlib.colors as mcolors
import torchvision.transforms as T

import torch.distributed as dist
from typing import Union, Tuple, Any, Callable, Iterator, Set, Optional, overload, TypeVar, Mapping, Dict, List
# import evaluate
# from mmseg.evaluation.metrics import IoUMetric
from torchmetrics import JaccardIndex 

# https://github.com/NielsRogge/Transformers-Tutorials/blob/master/DINOv2/Train_a_linear_classifier_on_top_of_DINOv2_for_semantic_segmentation.ipynb

class SupervisedSegmentation(ModelBaseS):
    def __init__(self, 
        ignore_index=255,
        **kwargs):
        super(SupervisedSegmentation, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        
        self.metrics = ['loss']
        self.metrics_log = [True]
        self.metrics_save = [False]

        self.backbone = self.networks.backbone()
        self.head = self.networks.discriminator()
        self.mIoU = JaccardIndex(task='multiclass', num_classes=self.num_classes, average='macro', ignore_index=ignore_index)
        # self.manual_metrics = evaluate.load("mean_iou")
        # self.manual_metrics = IoUMetric(ignore_index = 255, iou_metrics='mIoU')

    def forward(self, x):
        if hasattr(self.backbone, 'model'):
            z = self.backbone.model.forward_features(x)['x_norm_patchtokens']
        else:
            z = self.backbone(x)

        # patch_tokens = z[0][0]
        # patch_tokens = patch_tokens.reshape(-1, patch_tokens.shape[-1])
        y_hat = self.head(z)
        return z, y_hat

    def model_step(self, batch, stage='fit'):
        [x], y, idx = batch
        z, y_hat = self.forward(x)
        # y_hat = y_hat.reshape(y.shape[0], -1, y_hat.shape[-1])
        # y_hat = y_hat.reshape(y_hat.shape[0], int(y_hat.shape[1]**0.5), int(y_hat.shape[1]**0.5), y_hat.shape[2])
        if False:
            pb()
            visualize_mask(y_hat['pred_masks'].argmax(dim=1)[0]).show()

        if not isinstance(y_hat, list):
            y_hat = torch.nn.functional.interpolate(y_hat, size=(y.shape[1], y.shape[2]), mode="bilinear", align_corners=False)
        loss = self.loss(y_hat, y.long())

        if not isinstance(y_hat, list):
            self.mIoU.update(y_hat, y.long())
        else:
            pb()

        # predicted = y_hat.argmax(dim=1)

        # if stage != 'fit':
        #     self.manual_metrics.add_batch(
        #         predictions=predicted.detach().cpu().numpy(), 
        #         references=y.detach().cpu().numpy()
        #     )

        # y_hat_prob = torch.nn.Softmax(dim=1)(y_hat)
        # y_hat_prob = y_hat_prob.permute(0, 2, 3, 1)
        # y_hat_prob = y_hat_prob.reshape(-1, y_hat_prob.shape[-1])
        # y = y.reshape(-1)
        # =====================================
        results_dict = {
            'loss': loss.mean()
        }
        # =====================================
        return results_dict

    def predict_step(self, batch: Any, batch_idx: int):
        [x], y, idx = batch
        z = self.backbone(x)
        y_hat = self.head(z)
        return y_hat

# =====================================
    def log_metrics(self, prefix):
        mIoU = self.mIoU.compute()
        self.log(prefix+"/mIoU", mIoU, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.mIoU.reset()

    def on_validation_epoch_start( self ):
        super(SupervisedSegmentation, self).on_validation_epoch_start()
        self.log_metrics('train')

    def on_validation_epoch_end( self ):
        super(SupervisedSegmentation, self).on_validation_epoch_end()
        self.log_metrics('val')
        
    def on_test_epoch_end( self ):
        super(SupervisedSegmentation, self).on_test_epoch_end()
        self.log_metrics('test')

# ========================================
def visualize_mask(img):
    integer_image = img.cpu().clone().detach()

    # Determine the number of unique classes
    num_classes = torch.unique(integer_image).numel()
    num_classes = integer_image.max() + 1
    # pb()

    # Generate color palette in HSV and convert to RGB
    hues = np.linspace(0, 1, num_classes, endpoint=False)
    palette_hsv = [(hue, 1, 1) for hue in hues]  # Full saturation and value
    palette_rgb = [mcolors.hsv_to_rgb(color) for color in palette_hsv]
    palette_rgb = (np.array(palette_rgb) * 255).astype(np.uint8)

    # Map the integer image to an RGB image using the palette
    rgb_image = np.array(palette_rgb[integer_image.numpy()])

    # Convert the RGB image to a PIL image
    to_pil_image = T.ToPILImage()
    pil_image = to_pil_image(torch.ByteTensor(rgb_image).permute(2, 0, 1))
    return pil_image
