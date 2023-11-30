import os
from typing import Dict, Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR

import pytorch_lightning as pl

from config_settings import config_settings
from optimizers_schedulers import make_optimizer_scheduler
from loss.coarse_loss import CoarseLoss
from loss.refined_loss import RefinedLoss
from metrics.average_meter import AverageMeter

# from utils.vis_utils import save_depth_as_uint16png_upload, save_depth_as_uint8colored
# from utils.logger import logger


class LightningBaseModel(pl.LightningModule):
    def __init__(self, settings: config_settings):
        super().__init__()
        self.settings = settings

        self.depth_coarse_criterion = CoarseLoss(settings)
        self.depth_refined_criterion = RefinedLoss(settings)

        self.refined_average_meter = AverageMeter()
        self.test_average_meter = AverageMeter()

        # self.mylogger = logger(settings)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        model_bone_params = [p for _, p in self.named_parameters() if p.requires_grad]
        optimizer, scheduler = make_optimizer_scheduler(self.settings, model_bone_params)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def forward(self, input):
        pass

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch = self.forward(batch)

        gt = batch["gt"]
        pred_full = batch["pred_full"]
        pred_1_2 = batch["pred_1_2"]
        pred_1_4 = batch["pred_1_4"]
        pred_1_8 = batch["pred_1_8"]
        pred_1_16 = batch["pred_1_16"]

        coarse_loss = self.depth_coarse_criterion(gt, pred_1_2, pred_1_4, pred_1_8, pred_1_16)
        refined_loss = self.depth_refined_criterion(gt, pred_full)

        if self.current_epoch < self.settings.train_stage0:
            loss = 0.6 * refined_loss + 0.4 * coarse_loss
        elif self.current_epoch < self.settings.train_stage1:
            loss = 0.9 * refined_loss + 0.1 * coarse_loss
        else:
            loss = refined_loss

        self.log("train/loss", loss.item(), sync_dist=True)
        self.log("train/coarse_loss", coarse_loss.item(), sync_dist=True)
        self.log("train/refined_loss", refined_loss.item(), sync_dist=True)

        return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        batch = self.forward(batch)

        gt = batch["gt"]
        pred_full = batch["pred_full"]

        self.refined_average_meter.update(gt, pred_full)

        self.log("val/rmse_step", self.refined_average_meter.rmse, on_step=True, sync_dist=True)

        return

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().test_step(*args, **kwargs)

    # def training_epoch_end(self, *args: Any, **kwargs: Any) -> None:
    #     pass

    def on_validation_epoch_end(self) -> None:
        self.refined_average_meter.compute()
        self.log("val/rmse_epoch", self.refined_average_meter.sum_rmse, on_epoch=True, sync_dist=True)
        self.refined_average_meter.reset_all()

    # def on_test_epoch_end(self) -> None:
    #     pass

    def on_after_backward(self) -> None:
        """
        Skipping updates in case of unstable gradients
        https://github.com/Lightning-AI/lightning/issues/4956
        """
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break
        if not valid_gradients:
            print("detected inf or nan values in gradients. not updating model parameters")
            self.zero_grad()
