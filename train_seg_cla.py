import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_seg_cla_dataset
from model import ResUNet
from utils import create_dir, parse_arg, get_consistency_weight
from utils import dice_loss, compute_metric

create_dir()


def train_seg_cla(args):
    # prepare train and validation dataset
    train_set, val_set = get_seg_cla_dataset(args.data_path, args.train_val_ratio, args.labeled_ratio)

    # prepare dataloader
    train_loader = DataLoader(train_set, args.batch_size, True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, args.batch_size, True, num_workers=args.num_worker)

    # initialize student-teacher network
    net_student = ResUNet().to(args.device)
    net_teacher = ResUNet().to(args.device)
    net_teacher.requires_grad_(False)

    # define loss
    criterion_dice = dice_loss
    criterion_ce = torch.nn.CrossEntropyLoss()
    criterion_con = torch.nn.MSELoss()

    # define optimizer
    optim = torch.optim.Adam(net_student.parameters(), lr=args.lr)

    print('start training!')
    for epoch in range(args.epoch):

        # ####################################### train model #######################################
        loss_seg_history = []
        loss_con_history = []
        loss_cla_history = []

        for data, mask, label, is_labeled in tqdm(train_loader, desc='training progress', leave=False):
            data, mask, label = data.to(args.device), mask.to(args.device), label.to(args.device)

            # separate the data and mask into labeled and unlabeled parts
            idx_labeled = torch.where(is_labeled == 1)
            idx_unlabeled = torch.where(is_labeled == 0)

            # compute network prediction
            out_seg, out_cla = net_student.seg_cla_forward(data)

            # compute segmentation loss
            loss_seg = 0
            if len(idx_labeled) > 0:
                # compute segmentation loss
                loss_seg = criterion_dice(out_seg[idx_labeled], mask[idx_labeled])
                loss_seg = loss_seg + criterion_ce(out_seg[idx_labeled], mask[idx_labeled])
                loss_seg = loss_seg / len(idx_labeled)
                loss_seg_history.append(loss_seg.cpu().data.numpy())

            # compute classification loss
            loss_cla = criterion_ce(out_cla, label)
            loss_cla = loss_cla / len(data)
            loss_cla_history.append(loss_cla.cpu().data.numpy() * args.cla_weight)

            # compute consistency loss
            consistency_weight = get_consistency_weight(epoch)
            loss_con = 0
            if len(idx_unlabeled) > 0:
                out_stu = out_seg[idx_unlabeled]
                out_tea = net_teacher.noisy_forward(data[idx_unlabeled])

                loss_con = criterion_con(out_stu, out_tea) / len(idx_unlabeled)
                loss_con_history.append(loss_con.cpu().data.numpy() * consistency_weight)

            # combine the segmentation loss and the consistency loss
            loss = loss_seg + args.cla_weight * loss_cla + consistency_weight * loss_con

            # backward propagation and parameter update
            optim.zero_grad()
            loss.backward()
            optim.step()

            # update the teacher model
            param_student = net_student.state_dict()
            param_teacher = net_teacher.state_dict()

            # using moving exponential average to update teacher model
            for para_stu, (key_tea, para_tea) in zip(param_student.values(), param_teacher.items()):
                param_teacher[key_tea] = args.alpha * para_tea + (1 - args.alpha) * para_stu

            net_teacher.load_state_dict(param_teacher)

        loss_seg_history = float(np.mean(loss_seg_history))
        loss_cla_history = float(np.mean(loss_cla_history))
        loss_con_history = float(np.mean(loss_con_history))
        print('epoch: %d/%d | train | dice: %.4f | classification: %.4f | consistency: %.4f' % (
            epoch, args.epoch, loss_seg_history, loss_cla_history, loss_con_history))

        torch.save(net_student.state_dict(), './model/cla/net_%d.pth' % epoch)

        # ####################################### validate model #######################################

        # performance metrics
        pa = pa_total = 0
        iou = iou_total = 0
        dice = dice_total = 0
        acc = acc_total = 0

        with torch.no_grad():
            for data, mask, label, _ in tqdm(val_loader, desc='validation progress', leave=False):
                data = data.to(args.device)
                mask = mask.to(args.device)
                label = label.to(args.device)

                # network predict
                out_seg, out_cla = net_student.seg_cla_forward(data)
                out_seg = torch.argmax(out_seg, dim=1)

                out_cla = torch.argmax(out_cla, dim=1)
                acc += torch.sum(out_cla == label)
                acc_total += len(label)

                result = compute_metric(out_seg, mask)
                pa += result[0]
                iou += result[1]
                dice += result[2]
                pa_total += len(mask)
                iou_total += len(mask)
                dice_total += len(mask)

        print('epoch: %d/%d | val | DICE: %.3f | PA: %.3f | IOU: %.3f | ACC: %.3f' % (
            epoch, args.epoch, dice / dice_total, pa / pa_total, iou / iou_total, acc / acc_total))


if __name__ == '__main__':
    args = parse_arg()
    train_seg_cla(args)
