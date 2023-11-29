import torch
import torch.nn as nn
import torch.nn.functional as F

from loss.l1loss import L1Loss
from loss.l2loss import L2Loss


class NetLoss(nn.Module):
    def __init__(self, max_depth, decay, alpha, beta):
        """
        loss =          alpha * L1Loss(gt, pred_full) + beta * L2Loss(gt, pred_full) +
               decay * (alpha * L1Loss(gt, pred_1_2) + beta * L2Loss(gt, pred_1_2)) +
               decay^2 * (alpha * L1Loss(gt, pred_1_4) + beta * L2Loss(gt, pred_1_4)) +
               decay^3 * (alpha * L1Loss(gt, pred_1_8) + beta * L2Loss(gt, pred_1_8)) +
               decay^4 * (alpha * L1Loss(gt, pred_1_16) + beta * L2Loss(gt, pred_1_16))
        """
        super().__init__()

        self.l1 = L1Loss(max_depth)
        self.l2 = L2Loss(max_depth)

        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    def forward(self, gt, pred_full, pred_1_2, pred_1_4, pred_1_8, pred_1_16):
        gt_internal = gt.clone()
        loss = self.alpha * self.l1(gt_internal, pred_full) + self.beta * self.l2(gt_internal, pred_full)

        gt_internal = F.interpolate(gt_internal, scale_factor=0.5, mode="nearest")
        loss += self.decay * (self.alpha * self.l1(gt_internal, pred_1_2) + self.beta * self.l2(gt_internal, pred_1_2))

        gt_internal = F.interpolate(gt_internal, scale_factor=0.5, mode="nearest")
        loss += self.decay**2 * (self.alpha * self.l1(gt_internal, pred_1_4) + self.beta * self.l2(gt_internal, pred_1_4))

        gt_internal = F.interpolate(gt_internal, scale_factor=0.5, mode="nearest")
        loss += self.decay**3 * (self.alpha * self.l1(gt_internal, pred_1_8) + self.beta * self.l2(gt_internal, pred_1_8))

        gt_internal = F.interpolate(gt_internal, scale_factor=0.5, mode="nearest")
        loss += self.decay**4 * (self.alpha * self.l1(gt_internal, pred_1_16) + self.beta * self.l2(gt_internal, pred_1_16))

        return loss
