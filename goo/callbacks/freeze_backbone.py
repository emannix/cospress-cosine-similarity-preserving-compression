from lightning.pytorch.callbacks import BaseFinetuning
from typing import Any, Callable, Dict, Generator, Iterable, List, Optional, Union

import torch
from torch.nn import Module, ModuleDict
from torch.nn.modules.batchnorm import _BatchNorm
from torch.optim.optimizer import Optimizer

from lightning.pytorch.utilities.rank_zero import rank_zero_warn

from pdb import set_trace as pb

class FeatureExtractorFreezeUnfreeze(BaseFinetuning):
    def __init__(self, unfreeze_at_epoch=10, freeze_target='backbone'):
        super().__init__()
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.freeze_target = freeze_target

    # def setup(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str) -> None:
    #     super().setup(trainer, pl_module, stage)
    #     self.model = pl_module

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        self.freeze(getattr(pl_module, self.freeze_target), train_bn=False)

    @staticmethod
    def unfreeze_and_add_param_group(
        model,
        modules: Union[Module, Iterable[Union[Module, Iterable]]],
        optimizer: Optimizer,
        train_bn: bool = True,
    ) -> None:
        """Unfreezes a module and adds its parameters to an optimizer.

        Args:
            modules: A module or iterable of modules to unfreeze.
                Their parameters will be added to an optimizer as a new param group.
            optimizer: The provided optimizer will receive new parameters and will add them to
                `add_param_group`
            lr: Learning rate for the new param group.
            initial_denom_lr: If no lr is provided, the learning from the first param group will be used
                and divided by `initial_denom_lr`.
            train_bn: Whether to train the BatchNormalization layers.
        """
        BaseFinetuning.make_trainable(modules)
        params = FeatureExtractorFreezeUnfreeze.filter_params(modules, train_bn=train_bn, requires_grad=True)
        params = FeatureExtractorFreezeUnfreeze.filter_on_optimizer(optimizer, params)
        params = model.hparams.parameter_groups(model, params)

        params_lr = optimizer.param_groups[0]["lr"]
        params_lr_init = optimizer.param_groups[0]["initial_lr"]
        if params:
            for param in params:
                param['lr'] = params_lr
                param['initial_lr'] = params_lr_init
                optimizer.add_param_group(param)


    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        if current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                model=pl_module,
                modules=getattr(pl_module, self.freeze_target),
                optimizer=optimizer,
                train_bn=True,
            )

# ====================================

    @staticmethod
    def filter_params(
        modules: Union[Module, Iterable[Union[Module, Iterable]]], train_bn: bool = True, requires_grad: bool = True
    ) -> Generator:
        """Yields the `requires_grad` parameters of a given module or list of modules.

        Args:
            modules: A given module or an iterable of modules
            train_bn: Whether not to train the BatchNorm module
            requires_grad: Whether to create a generator for trainable or non-trainable parameters.
        Returns:
            Generator
        """
        modules = BaseFinetuning.flatten_modules(modules)
        for mod in modules:
            if isinstance(mod, _BatchNorm) and not train_bn:
                continue
            # recursion could yield duplicate parameters for parent modules w/ parameters so disabling it
            for n, param in mod.named_parameters(recurse=False):
                if param.requires_grad == requires_grad:
                    yield n, param

    @staticmethod
    def filter_on_optimizer(optimizer: Optimizer, params: Iterable) -> List:
        """This function is used to exclude any parameter which already exists in this optimizer.

        Args:
            optimizer: Optimizer used for parameter exclusion
            params: Iterable of parameters used to check against the provided optimizer

        Returns:
            List of parameters not contained in this optimizer param groups
        """
        out_params = []
        removed_params = []
        for n, param in params:
            if not any(torch.equal(p, param) for group in optimizer.param_groups for p in group["params"]):
                out_params.append((n, param))
            else:
                removed_params.append(param)

        if removed_params:
            rank_zero_warn(
                "The provided params to be frozen already exist within another group of this optimizer."
                " Those parameters will be skipped.\n"
                "HINT: Did you init your optimizer in `configure_optimizer` as such:\n"
                f" {type(optimizer)}(filter(lambda p: p.requires_grad, self.parameters()), ...) ",
            )
        return out_params