import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir
from tqdm import tqdm

create_dir()


def train_semi():
    # hyper-parameters
    BATCH_SIZE = 128
    SUPERVISED_RATIO = 0.5
    LR = 1e-3
    EPOCH = 50
    DEVICE = torch.device('cpu')
    PATH = '.'

    # preparing dataset
    train_set = OxfordIIITPetSeg(PATH + '/data', split='trainval', download=True)
    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=2)
    test_set = OxfordIIITPetSeg(PATH + '/data', split='test', download=True)
    test_loader = DataLoader(test_set, int(BATCH_SIZE * SUPERVISED_RATIO), True, num_workers=16)

    # initialize network
    net_student = ResUNet().to(DEVICE)
    net_teacher = ResUNet().to(DEVICE)

    # define loss and optimizer
    criterion = torch.nn.MSELoss()
    optim = torch.optim.Adam(net_student.parameters(), lr=LR)

    print('start training!')
    for epoch in range(EPOCH):

        # ####################################### train model #######################################
        loss_history = []

        # for data, mask in train_loader:
        for data_labeled, mask_labeled, data_unlabeled in tqdm(train_loader, desc='training progress', leave=False):

            # separate the data into labeled and unlabeled parts
            data_labeled = data_labeled.to(DEVICE)
            mask_labeled = mask_labeled.to(DEVICE)
            data_unlabeled = data_unlabeled.to(DEVICE)

            # compute supervised loss
            out_student = net_student(data_labeled)
            loss_segmentation = criterion(out_student, mask_labeled)

            # compute consistency loss
            out_teacher = net_teacher(data_unlabeled)
            loss_consistency = criterion(out_student, out_teacher)

            # backward propagation and parameter update
            optim.zero_grad()
            loss_segmentation.backward()
            optim.step()

            loss_history.append(loss_segmentation.cpu().data.numpy())

        print('epoch: %d | loss: %.3f' % (epoch, float(np.mean(loss_history))))

        torch.save(net_student.state_dict(), PATH + '/model/supervised/net_%d.pth' % epoch)

        continue
        # ####################################### test model #######################################

        # performance metrics
        PA = PA_TOTAL = 0
        IOU = IOU_TOTAL = 0

        with torch.no_grad():
            for data, mask in tqdm(test_loader, desc='testing progress', leave=False):
                data, mask = data.to(DEVICE), mask.to(DEVICE)

                # network predict
                out_student = net_student(data)

                out_student[out_student >= 0.5] = 1
                out_student[out_student < -0.5] = -1
                out_student[abs(out_student) != 1] = 0

                # compute the pixel accuracy metric
                PA += torch.sum(out_student == mask)
                PA_TOTAL += np.cumprod(mask.shape)[-1]

                # compute the IOU metric
                out_student[out_student != 1] = 0
                mask[mask != 1] = 0

                region_intersection = torch.sum(out_student * mask > 0, dim=(1, 2, 3))
                region_union = torch.sum(out_student + mask > 0, dim=(1, 2, 3))

                IOU += torch.sum(region_intersection / region_union)
                IOU_TOTAL += len(mask)

        print('epoch: %d | PA: %.3f | IOU: %.3f' % (epoch, PA / PA_TOTAL, IOU / IOU_TOTAL))


if __name__ == '__main__':
    train_semi()
