
from .model_base_s import ModelBaseS
from pdb import set_trace as pb

from torchmetrics.functional.classification import multiclass_accuracy
from torch.nn.functional import softmax
import torch

class Supervised(ModelBaseS):
    def __init__(self, track_norms = False, **kwargs):
        super(Supervised, self).__init__(**kwargs)
        self.save_hyperparameters(logger=False)
        
        self.metrics = ['index', 'label', 'y_hat', 'loss_vec', 'y_hat_log']
        self.metrics_log = [False, False, False, False, False]
        self.metrics_save = [True, True, True, True, True]
        if track_norms:
            self.metrics = self.metrics + ['final_layer_norm', 'bias_val', 'activation_norm']
            self.metrics_log = self.metrics_log + [True, True, True]
            self.metrics_save = self.metrics_save + [False, False, False]

        self.backbone = self.networks.backbone()
        self.discriminator = self.networks.discriminator()

    def forward(self, x):
        z = self.backbone(x)
        y_hat = self.discriminator(z)
        return y_hat, z

    def model_step(self, batch, stage='fit'):
        [x], y, idx = batch
        y_hat, z = self.forward(x)
        loss = self.loss(y_hat, y)
        # =====================================
        y_hat_prob = softmax(y_hat, dim=1)
        # acc = multiclass_accuracy(y_hat_prob, y, self.num_classes)
        results_dict = {
            'loss': loss.mean(), 'index': idx, 'label': y, 'y_hat': y_hat_prob, 'loss_vec': loss, 'y_hat_log': y_hat #, 'acc': acc
        }
        if self.hparams.track_norms:
            results_dict['activation_norm'] = z.norm(dim=1).mean()
            results_dict['final_layer_norm'] = self.discriminator.fc.weight.norm(dim=1).mean()
            if self.discriminator.fc.bias is None:
                bias_norm = torch.tensor(0.0)
            else:
                bias_norm = self.discriminator.fc.bias.mean()
            results_dict['bias_val'] = bias_norm

        # =====================================
        return results_dict
        
    def predict_step(self, batch, stage='fit'):
        [x], y, idx = batch
        y_hat, z = self.forward(x)
        return y_hat