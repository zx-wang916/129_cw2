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
    BATCH_SIZE = 32
    LABELED_RATIO = 0.5
    LR = 1e-3
    EPOCH = 50
    ALPHA = 0.9
    CONSISTENCY_WEIGHT = 1
    DEVICE = torch.device('cuda:4')
    # DEVICE = torch.device('cpu')
    NUM_WORKERS = 8
    PATH = '.'

    # preparing dataset
    train_set = OxfordIIITPetSeg(PATH + '/data', train=True, labeled_ratio=LABELED_RATIO)
    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    val_set = OxfordIIITPetSeg(PATH + '/data', train=False)
    val_loader = DataLoader(val_set, BATCH_SIZE, True, num_workers=NUM_WORKERS)

    # initialize network
    net_student = ResUNet()
    net_student = net_student.to(DEVICE)
    net_teacher = ResUNet()
    net_teacher = net_teacher.to(DEVICE)

    # define loss and optimizer
    criterion_seg = dice_loss
    criterion_con = torch.nn.MSELoss()
    optim = torch.optim.Adam(net_student.parameters(), lr=LR)

    print('start training!')
    for epoch in range(EPOCH):

        # ####################################### train model #######################################
        loss_history = []

        # for data, mask in train_loader:
        for data, mask, unlabeled in tqdm(train_loader, desc='training progress', leave=False):

            # separate the data and mask into labeled and unlabeled parts
            data, mask, unlabeled = data.to(DEVICE), mask.to(DEVICE), unlabeled.to(DEVICE)

            # compute segmentation loss
            out = net_student(data)
            loss_seg = criterion_seg(out, mask)

            # compute consistency loss
            noise_tea = torch.normal(0, 0.1, unlabeled.shape).to(DEVICE)
            out_stu = net_student(unlabeled)
            out_tea = net_teacher(unlabeled + noise_tea)
            loss_con = criterion_con(out_stu, out_tea)

            # combine the segmentation loss and the consistency loss
            loss = loss_seg + CONSISTENCY_WEIGHT * loss_con

            # backward propagation and parameter update
            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the teacher model
            param_student = net_student.state_dict()
            param_teacher = net_teacher.state_dict()

            for para_stu, (key_tea, para_tea) in zip(param_student.values(), param_teacher.items()):
                para_tea = ALPHA * para_tea + (1 - ALPHA) * para_stu
                param_teacher[key_tea] = para_tea

            net_teacher.load_state_dict(param_teacher)

            loss_history.append(loss_seg.cpu().data.numpy())

        print('epoch: %d | train | dice loss: %.3f' % (epoch, float(np.mean(loss_history))))

        torch.save(net_student.state_dict(), PATH + '/model/supervised/net_%d.pth' % epoch)

        # ####################################### validate model #######################################

        # performance metrics
        PA = PA_TOTAL = 0
        IOU = IOU_TOTAL = 0

        with torch.no_grad():
            for data, mask in tqdm(val_loader, desc='validation progress', leave=False):
                data, mask = data.to(DEVICE), mask.to(DEVICE)

                # network predict
                out = net_student(data)
                out = torch.argmax(out, dim=1)

                # compute the pixel accuracy metric
                mask_pa = torch.argmax(mask, dim=1)
                PA += torch.sum(out == mask_pa)
                PA_TOTAL += np.cumprod(mask_pa.shape)[-1]

                for i in range(3):
                    # compute binary mask for segmentation of each class
                    out_class_i = torch.zeros_like(out)
                    out_class_i[torch.where(out == i)] = 1
                    mask_class_i = mask[:, i]

                    # compute the IOU metric
                    region_intersection = torch.sum(out_class_i * mask_class_i > 0, dim=(1, 2))
                    region_union = torch.sum(out_class_i + mask_class_i > 0, dim=(1, 2))

                    IOU += torch.sum(region_intersection / region_union)
                    IOU_TOTAL += len(mask)

        print('epoch: %d | val | PA: %.3f | IOU: %.3f' % (epoch, PA / PA_TOTAL, IOU / IOU_TOTAL))


if __name__ == '__main__':
    train_supervised()
