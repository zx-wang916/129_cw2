import os
import torch
import numpy as np


def create_dir():
    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isdir('model'):
        os.mkdir('model')

    if not os.path.isdir('model/supervised'):
        os.mkdir('model/supervised')

    if not os.path.isdir('model/semi'):
        os.mkdir('model/semi')


def dice_loss(out, target):
    loss = 2 * torch.sum(out * target) + 1
    loss /= torch.sum(out) + torch.sum(target) + 1
    loss = 1 - loss
    return loss


# metrics for evaluating the segmentation performance

def compute_region(pred, target):
    # compute the TP, FP, TN, FN region area
    # input should be the shape (batch, H, W)
    TP = torch.sum(pred * target, dim=(1, 2))
    FP = torch.sum(pred * (1 - target), dim=(1, 2))
    TN = torch.sum((1 - pred) * (1 - target), dim=(1, 2))
    FN = torch.sum((1 - pred) * target, dim=(1, 2))
    return TP, FP, TN, FN


def metric_dice(TP, FP, TN, FN):
    return 2 * TP / (FP + 2 * TP + FN + 1e-10)


def metric_pa(TP, FP, TN, FN):
    # pixel accuracy
    return (TP + TN) / (TP + FP + TN + FN + 1e-10)


def metric_IOU(TP, FP, TN, FN):
    return TP / (FP + TP + FN + 1e-10)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    consistency = 5
    rampup_length = 50

    if rampup_length == 0:
        weight = 1.0
    else:
        epoch = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - epoch / rampup_length
        weight = float(np.exp(-5.0 * phase * phase))
    return consistency * weight
