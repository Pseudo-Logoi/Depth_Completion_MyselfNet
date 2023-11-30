import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.l1loss import L1Loss
from loss.l2loss import L2Loss

from config_settings import config_settings


class RefinedLoss(nn.Module):
    def __init__(self, settings: config_settings):
        """
        loss = alpha * L1Loss(gt, pred_full) + beta * L2Loss(gt, pred_full)
        """
        super().__init__()

        self.settings = settings
        self.max_depth = settings.max_depth
        self.decay = settings.decay
        self.alpha = settings.alpha
        self.beta = settings.beta

        self.l1 = L1Loss(self.max_depth)
        self.l2 = L2Loss(self.max_depth)

    def forward(self, gt, pred_full):
        loss = self.alpha * self.l1(gt, pred_full) + self.beta * self.l2(gt, pred_full)
        return loss
