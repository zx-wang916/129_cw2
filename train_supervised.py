import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir, dice_loss
from tqdm import tqdm

create_dir()


def train_supervised():
    # hyper-parameters
    BATCH_SIZE = 128
    LABELED_RATIO = 0.5
    LR = 1e-3
    EPOCH = 50
    ALPHA = 0.5
    DEVICE = torch.device('cuda:2')
    NUM_WORKERS = 8
    PATH = '.'

    # preparing dataset
    train_set = OxfordIIITPetSeg(PATH + '/data', 'trainval', download=True)
    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    test_set = OxfordIIITPetSeg(PATH + '/data', 'test', download=True)
    test_loader = DataLoader(test_set, BATCH_SIZE, True, num_workers=NUM_WORKERS)

    # initialize network
    net = ResUNet()
    net = net.to(DEVICE)

    # define loss and optimizer
    # criterion = torch.nn.MSELoss()
    criterion = dice_loss
    optim = torch.optim.Adam(net.parameters(), lr=LR)

    print('start training!')
    for epoch in range(EPOCH):

        # ####################################### train model #######################################
        loss_history = []

        # for data, mask in train_loader:
        for data, mask in tqdm(train_loader, desc='training progress', leave=False):

            # separate the data and mask into labeled and unlabeled parts
            data, mask = data.to(DEVICE), mask.to(DEVICE)
            idx_labeled = int(len(data) * LABELED_RATIO)
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

        print('epoch: %d | dice loss: %.3f' % (epoch, float(np.mean(loss_history))))

        torch.save(net.state_dict(), PATH + '/model/supervised/net_%d.pth' % epoch)

        # ####################################### test model #######################################

        # performance metrics
        PA = PA_TOTAL = 0
        IOU = IOU_TOTAL = 0

        with torch.no_grad():
            for data, mask in tqdm(test_loader, desc='testing progress', leave=False):
                data, mask = data.to(DEVICE), mask.to(DEVICE)

                # network predict
                out = net(data)
                out = torch.argmax(out, dim=1, keepdim=True)

                # compute the pixel accuracy metric
                PA += torch.sum(out == mask)
                PA_TOTAL += np.cumprod(mask.shape)[-1]

                # compute the IOU metric
                for i in range(3):
                    out_class_i = torch.zeros_like(mask)
                    out_class_i[torch.where(out == i)] = 1
                    mask_class_i = torch.zeros_like(mask)
                    mask_class_i[torch.where(mask == i)] = 1

                    region_intersection = torch.sum(out_class_i * mask_class_i > 0, dim=(1, 2, 3))
                    region_union = torch.sum(out_class_i + mask_class_i > 0, dim=(1, 2, 3))

                    IOU += torch.sum(region_intersection / region_union)
                    IOU_TOTAL += len(mask)

        print('epoch: %d | PA: %.3f | IOU: %.3f' % (epoch, PA / PA_TOTAL, IOU / IOU_TOTAL))


if __name__ == '__main__':
    train_supervised()
