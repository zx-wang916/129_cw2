import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir, dice_loss, compute_region, metric_dice, metric_iou, metric_pa, get_consistency_weight
from tqdm import tqdm

create_dir()


def train_supervised():
    # hyper-parameters
    BATCH_SIZE = 64
    TRAIN_VAL_RATIO = 0.8
    LABELED_RATIO = 0.1
    LR = 1e-3
    EPOCH = 300
    ALPHA = 0.99
    DEVICE = torch.device('cuda:5')
    # DEVICE = torch.device('cpu')
    NUM_WORKERS = 8
    WORKSPACE_PATH = '.'

    # prepare train and validation dataset
    train_set, val_set = OxfordIIITPetSeg.split_train_val(
        WORKSPACE_PATH + '/data', TRAIN_VAL_RATIO, LABELED_RATIO)

    # prepare dataloader
    train_loader = DataLoader(train_set, BATCH_SIZE, True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_set, BATCH_SIZE, True, num_workers=NUM_WORKERS)

    # initialize student-teacher network
    net_student = ResUNet()
    net_student = net_student.to(DEVICE)
    net_teacher = ResUNet()
    net_teacher = net_teacher.to(DEVICE)
    net_teacher.requires_grad_(False)

    # define loss
    criterion_dice = dice_loss
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_con = torch.nn.MSELoss()

    # define optimizer
    optim = torch.optim.Adam(net_student.parameters(), lr=LR)

    print('start training!')
    for epoch in range(EPOCH):

        # ####################################### train model #######################################
        loss_seg_history = []
        loss_con_history = []

        for data, mask, is_labeled in tqdm(train_loader, desc='training progress', leave=False):

            # separate the data and mask into labeled and unlabeled parts
            data, mask = data.to(DEVICE), mask.to(DEVICE)

            data_labeled = data[torch.where(is_labeled == 1)]
            mask_labeled = mask[torch.where(is_labeled == 1)]
            data_unlabeled = data[torch.where(is_labeled == 0)]

            # compute segmentation loss
            loss_seg = 0
            if len(data_labeled) > 0:
                # predict
                out = net_student(data_labeled)

                # compute dice loss
                loss_seg = criterion_dice(out, mask_labeled)

                # compute cross-entropy loss
                loss_seg = loss_seg + criterion_ce(out, mask_labeled)

                loss_seg = loss_seg / len(data_labeled)
                loss_seg_history.append(loss_seg.cpu().data.numpy())

            # compute consistency loss
            consistency_weight = get_consistency_weight(epoch)
            loss_con = 0
            if len(data_unlabeled) > 0:
                out_stu = net_student.noisy_forward(data_unlabeled)
                out_tea = net_teacher.noisy_forward(data_unlabeled)
                loss_con = criterion_con(out_stu, out_tea) / len(data_unlabeled)
                loss_con_history.append(loss_con.cpu().data.numpy() * consistency_weight)

            # combine the segmentation loss and the consistency loss
            loss = loss_seg + consistency_weight * loss_con

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

        print('epoch: %d | train | dice loss: %.4f | consistency loss: %.4f' % (
            epoch, float(np.mean(loss_seg_history)), float(np.mean(loss_con_history))))

        torch.save(net_student.state_dict(), WORKSPACE_PATH + '/model/semi/net_%d.pth' % epoch)

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

                    # compute TP, TN, FP, FN
                    region = compute_region(out_class_i, mask_class_i)

                    # compute pixel accuracy
                    pa += torch.sum(metric_pa(*region))
                    pa_total += len(mask)

                    # compute IOU
                    iou += torch.sum(metric_iou(*region))
                    iou_total += len(mask)

                    # compute dice
                    dice += torch.sum(metric_dice(*region))
                    dice_total += len(mask)

        print('epoch: %d | val | DICE: %.3f | PA: %.3f | IOU: %.3f' % (
            epoch, dice / dice_total, pa / pa_total, iou / iou_total))


if __name__ == '__main__':
    train_supervised()
