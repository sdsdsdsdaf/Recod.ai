import torch
import torch.nn.functional as F

class SoftDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, y_pred, y_true):
        y_pred = torch.sigmoid(y_pred)  # if logits
        y_true = y_true.float()
        intersection = (y_pred * y_true).sum()
        denom = y_pred.pow(2).sum() + y_true.pow(2).sum()
        dice = (2. * intersection + self.smooth) / (denom + self.smooth)
        return 1 - dice

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class FocalTverskyLoss(nn.Module):
    """
    Wrapper around smp.losses.TverskyLoss to apply focal modulation.
    """
    def __init__(self, mode='binary', alpha=0.3, beta=0.7, gamma=0.75, smooth=1e-6, **kwargs):
        super().__init__()
        self.gamma = gamma
        self.base_loss = smp.losses.TverskyLoss(
            mode=mode,
            smooth=smooth,
            alpha=alpha,
            beta=beta,
            **kwargs,
        )

    def forward(self, y_pred, y_true):
        # smp.losses.TverskyLoss returns a tensor (mean loss)
        base_loss = self.base_loss(y_pred, y_true)
        # Apply focal modulation
        return torch.pow(base_loss, self.gamma)
