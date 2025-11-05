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
