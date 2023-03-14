import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir
create_dir()


BATCH_SIZE = 16
LR = 1e-3
EPOCH = 10
DEVICE = torch.device('cpu')

train_set = OxfordIIITPetSeg('./data', split='trainval', download=True)
train_loader = DataLoader(train_set, BATCH_SIZE, True)

net = ResUNet()
net = net.to(DEVICE)

criterion = torch.nn.MSELoss()
optim = torch.optim.Adam(net.parameters(), lr=LR)


print('start training!')
for epoch in range(EPOCH):

    loss_history = []
    for data, mask in train_loader:
        data, mask = data.to(DEVICE), mask.to(DEVICE)

        out = net(data)

        loss = criterion(out, mask)

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_history.append(loss.cpu().data.numpy())

    print('epoch: %d | loss: %.3f' % (epoch, float(np.mean(loss_history))))

    torch.save(net.cpu().state_dict(), 'model/supervised/net_%d.pth' % epoch)
