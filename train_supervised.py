import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir

create_dir()

# hyper-parameters
BATCH_SIZE = 64
SUPERVISED_RATIO = 0.5
LR = 1e-3
EPOCH = 10
DEVICE = torch.device('cpu')
PATH = '.'

# preparing dataset
train_set = OxfordIIITPetSeg(PATH + '/data', split='trainval', download=True)
train_loader = DataLoader(train_set, BATCH_SIZE, True)
test_set = OxfordIIITPetSeg(PATH + '/data', split='test', download=True)
test_loader = DataLoader(test_set, int(BATCH_SIZE * SUPERVISED_RATIO), True)

# initialize network
net = ResUNet()
net = net.to(DEVICE)

# define loss and optimizer
criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=LR)

print('start training!')
for epoch in range(EPOCH):

    # ####################################### train model #######################################

    loss_history = []
    for data, mask in train_loader:
        # separate the data into labeled and unlabeled parts
        data, mask = data.to(DEVICE), mask.to(DEVICE)
        idx_labeled = int(BATCH_SIZE * SUPERVISED_RATIO)
        data_labeled, mask_labeled = data[:idx_labeled], mask[:idx_labeled]

        # network predict
        out = net(data_labeled)

        # compute loss
        loss = criterion(out, mask_labeled)

        # backward propagation and parameter update
        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_history.append(loss.cpu().data.numpy())

    print('epoch: %d | loss: %.3f' % (epoch, float(np.mean(loss_history))))

    torch.save(net.state_dict(), PATH + '/model/supervised/net_%d.pth' % epoch)

    # ####################################### test model #######################################

    # performance metrics
    PA = PA_TOTAL = 0
    IOU = IOU_TOTAL = 0

    with torch.no_grad():
        for data, mask in test_loader:
            data, mask = data.to(DEVICE), mask.to(DEVICE)

            # network predict
            out = net(data)

            out[out >= 0.5] = 1
            out[out < -0.5] = -1
            out[abs(out) != 1] = 0

            # compute the pixel accuracy metric
            PA += torch.sum(out == mask)
            PA_TOTAL += np.cumprod(mask.shape)[-1]

            # compute the IOU metric
            out[out != 1] = 0
            mask[mask != 1] = 0
            region_intersection = torch.sum(out == mask, dim=(1, 2, 3))
            region_union = torch.sum(out + mask > 0, dim=(1, 2, 3))

            IOU += torch.sum(region_intersection / region_union)
            IOU_TOTAL += len(mask)

    print('epoch: %d | PA: %.3f | IOU: %.3f' % (epoch, PA / PA_TOTAL, IOU / IOU_TOTAL))
