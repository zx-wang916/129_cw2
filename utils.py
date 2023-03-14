import os

import torch


def create_dir():
    if not os.path.isdir('data'):
        os.mkdir('data')

    if not os.path.isdir('model'):
        os.mkdir('model')

    if not os.path.isdir('model/supervised'):
        os.mkdir('model/supervised')


def dice_loss(out, target):
    loss = 2 * torch.sum(out * target) + 1
    loss /= torch.sum(out) + torch.sum(target) + 1
    loss = 1 - loss
    return loss
