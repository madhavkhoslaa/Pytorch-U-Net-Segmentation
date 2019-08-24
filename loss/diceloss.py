import torch
import torch.nn as nn
import torch.functional as F


class Loss:
    @classmethod
    def dice_loss(input, target):
        smooth = 1.
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1 - ((2. * intersection + smooth)/(iflat.sum() + tflat.sum() + smooth))