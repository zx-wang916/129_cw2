import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


BATCH_SIZE = 16
DEVICE = torch.device('cpu')
MODEL_PATH = 'model/supervised/net_2.pth'

test_set = OxfordIIITPetSeg('./data', split='test', download=True)
test_loader = DataLoader(test_set, BATCH_SIZE, True)

net = ResUNet()
net = net.to(DEVICE)

net.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))

for data, mask in test_loader:
    data, mask = data.to(DEVICE), mask.to(DEVICE)

    out = net(data)

    _, ax = plt.subplots(1, 2)
    ax[0].imshow(data[0].permute(1, 2, 0))
    ax[1].imshow(out[0, 0].data.numpy() * 255, cmap='gray')
    plt.show()

    break

