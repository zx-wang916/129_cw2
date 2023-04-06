import os
import torch
import numpy as np
import argparse


def create_dir():
    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isdir('model'):
        os.mkdir('model')


def parse_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=64, required=False)
    parser.add_argument('-l', '--labeled_ratio', type=float, default=0.1, required=False)
    parser.add_argument('-u', '--unlabeled_ratio', type=float, default=1.0, required=False)
    parser.add_argument('-t', '--train_val_ratio', type=float, default=0.8, required=False)
    parser.add_argument('-con', '--consistency', type=float, default=100, required=False)
    parser.add_argument('-rl', '--rampup_len', type=float, default=100, required=False)
    parser.add_argument('-a', '--alpha', type=float, default=0.99, required=False)
    parser.add_argument('-c', '--cla_weight', type=float, default=0.3, required=False)
    parser.add_argument('-lr', '--lr', type=float, default=1e-3, required=False)
    parser.add_argument('-e', '--epoch', type=int, default=400, required=False)
    parser.add_argument('-d', '--device', type=str, default='cpu', required=False)
    parser.add_argument('-n', '--num_worker', type=int, default=8, required=False)
    parser.add_argument('-p', '--data_path', type=str, default='./data', required=False)
    return parser.parse_args()


def dice_loss(pred, mask, ep=1e-8):
    # metrics for evaluating the segmentation performance
    intersection = 2 * torch.sum(pred * mask, dim=(1, 2, 3)) + ep
    union = torch.sum(pred, dim=(1, 2, 3)) + torch.sum(mask, dim=(1, 2, 3)) + ep
    loss = 1 - intersection / union
    return torch.mean(loss)


def compute_metric(pred, mask):
    pa = iou = dice = 0

    for i in range(3):
        # compute binary mask for segmentation of each class
        out_class_i = torch.zeros_like(pred)
        out_class_i[torch.where(pred == i)] = 1
        mask_class_i = mask[:, i]

        region = compute_region(out_class_i, mask_class_i)

        pa += torch.sum(metric_pa(*region))
        iou += torch.sum(metric_iou(*region))
        dice += torch.sum(metric_dice(*region))

    return pa / 3, iou / 3, dice / 3


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


def metric_iou(TP, FP, TN, FN):
    return TP / (FP + TP + FN + 1e-10)


def get_consistency_weight(epoch, consistency=500, rampup_length=100):
    if rampup_length == 0:
        weight = 1.0
    else:
        epoch = np.clip(epoch, 0.0, rampup_length)
        phase = 1.0 - epoch / rampup_length
        weight = float(np.exp(-5.0 * phase * phase))
    return consistency * weight
