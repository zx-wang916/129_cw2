import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir, dice_loss, compute_region, metric_dice, metric_IOU, metric_pa, get_current_consistency_weight
from tqdm import tqdm

create_dir()


def train_supervised():
    # hyper-parameters
    BATCH_SIZE = 32
    LABELED_RATIO = 0.2
    LR = 1e-3
    EPOCH = 200
    ALPHA = 0.999
    CONSISTENCY_WEIGHT = 1
    DEVICE = torch.device('cuda:5')
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
        for data, mask, is_labeled in tqdm(train_loader, desc='training progress', leave=False):

            # separate the data and mask into labeled and unlabeled parts
            data, mask = data.to(DEVICE), mask.to(DEVICE)

            data_labeled = data[torch.where(is_labeled == 1)]
            mask_labeled = mask[torch.where(is_labeled == 1)]
            data_unlabeled = data[torch.where(is_labeled == 0)]

            # compute segmentation loss
            out = net_student(data_labeled)
            loss_seg = criterion_seg(out, mask_labeled) / len(data_labeled)

            # compute consistency loss
            noise_stu = torch.normal(0, 0.01, data_unlabeled.shape).to(DEVICE)
            noise_tea = torch.normal(0, 0.01, data_unlabeled.shape).to(DEVICE)
            out_stu = net_student(data_unlabeled + noise_stu)
            out_tea = net_teacher(data_unlabeled + noise_tea)
            loss_con = criterion_con(out_stu, out_tea) / len(data_unlabeled)

            # combine the segmentation loss and the consistency loss
            loss = loss_seg + get_current_consistency_weight(epoch) * loss_con

            # backward propagation and parameter update
            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the teacher model
            param_student = net_student.state_dict()
            param_teacher = net_teacher.state_dict()

            # using moving exponential average to update teacher model
            for para_stu, (key_tea, para_tea) in zip(param_student.values(), param_teacher.items()):
                mea = ALPHA * para_tea + (1 - ALPHA) * para_stu
                param_teacher[key_tea] = mea

            net_teacher.load_state_dict(param_teacher)

            loss_history.append(loss_seg.cpu().data.numpy())

        print('epoch: %d | train | dice loss: %.3f' % (epoch, float(np.mean(loss_history))))

        torch.save(net_student.state_dict(), PATH + '/model/semi/net_%d.pth' % epoch)

        # ####################################### validate model #######################################

        # performance metrics
        pa = pa_total = 0
        iou = iou_total = 0
        dice = dice_total = 0

        with torch.no_grad():
            for data, mask in tqdm(val_loader, desc='validation progress', leave=False):
                data, mask = data.to(DEVICE), mask.to(DEVICE)

                # network predict
                out = net_student(data)
                out = torch.argmax(out, dim=1)

                for i in range(3):
                    # compute binary mask for segmentation of each class
                    out_class_i = torch.zeros_like(out)
                    out_class_i[torch.where(out == i)] = 1
                    mask_class_i = mask[:, i]

                    region = compute_region(out_class_i, mask_class_i)

                    pa += torch.sum(metric_pa(*region))
                    pa_total += len(mask)

                    iou += torch.sum(metric_IOU(*region))
                    iou_total += len(mask)

                    dice += torch.sum(metric_dice(*region))
                    dice_total += len(mask)

        print('epoch: %d | val | DICE: %.3f | PA: %.3f | IOU: %.3f' % (
            epoch, dice / dice_total, pa / pa_total, iou / iou_total))


if __name__ == '__main__':
    train_supervised()
