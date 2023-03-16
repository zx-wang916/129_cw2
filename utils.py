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


def dice_loss(pred, mask, ep=1e-8):
    # metrics for evaluating the segmentation performance
    intersection = 2 * torch.sum(pred * mask) + ep
    union = torch.sum(pred) + torch.sum(mask) + ep
    loss = 1 - intersection / union
    return loss


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


def get_current_consistency_weight(epoch, consistency=50, rampup_length=100):
    if rampup_length == 0:
        weight = 1.0
    else:
        epoch = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - epoch / rampup_length
        weight = float(np.exp(-5.0 * phase * phase))
    return consistency * weight


if __name__ == '__main__':
    for i in range(200):
        print(get_current_consistency_weight(i))
