import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.l1loss import L1Loss
from loss.l2loss import L2Loss

from config_settings import config_settings


class CoarseLoss(nn.Module):
    def __init__(self, settings: config_settings):
        """
        loss = decay * (alpha * L1Loss(gt, pred_1_2) + beta * L2Loss(gt, pred_1_2)) +
               decay^2 * (alpha * L1Loss(gt, pred_1_4) + beta * L2Loss(gt, pred_1_4)) +
               decay^3 * (alpha * L1Loss(gt, pred_1_8) + beta * L2Loss(gt, pred_1_8)) +
               decay^4 * (alpha * L1Loss(gt, pred_1_16) + beta * L2Loss(gt, pred_1_16))
        """
        super().__init__()

        self.settings = settings
        self.max_depth = settings.max_depth
        self.decay = settings.decay
        self.alpha = settings.alpha
        self.beta = settings.beta

        self.l1 = L1Loss(self.max_depth)
        self.l2 = L2Loss(self.max_depth)

    def forward(self, gt, pred_1_2, pred_1_4, pred_1_8, pred_1_16):
        gt = F.interpolate(gt, scale_factor=0.5, mode="nearest")
        loss = self.decay * (self.alpha * self.l1(gt, pred_1_2) + self.beta * self.l2(gt, pred_1_2))

        gt = F.interpolate(gt, scale_factor=0.5, mode="nearest")
        loss += self.decay**2 * (self.alpha * self.l1(gt, pred_1_4) + self.beta * self.l2(gt, pred_1_4))

        gt = F.interpolate(gt, scale_factor=0.5, mode="nearest")
        loss += self.decay**3 * (self.alpha * self.l1(gt, pred_1_8) + self.beta * self.l2(gt, pred_1_8))

        gt = F.interpolate(gt, scale_factor=0.5, mode="nearest")
        loss += self.decay**4 * (self.alpha * self.l1(gt, pred_1_16) + self.beta * self.l2(gt, pred_1_16))

        return loss
