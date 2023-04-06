import numpy as np
import torch
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_seg_cla_dataset
from model import ResUNet
from utils import create_dir, parse_arg
from utils import dice_loss, compute_metric

create_dir()


def train_supervised(args):
    # prepare train and validation dataset
    train_set, val_set = get_seg_cla_dataset(args.data_path, args.train_val_ratio, args.labeled_ratio)

    # prepare dataloader
    train_loader = DataLoader(train_set, args.batch_size, True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, args.batch_size, True, num_workers=args.num_worker)

    # initialize network
    net = ResUNet().to(args.device)

    # define loss and optimizer
    criterion_dice = dice_loss
    criterion_ce = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    logging.info('start training!')
    best_dice = 0

    for epoch in range(args.epoch):

        # ####################################### train model #######################################

        loss_seg_history = []
        loss_cla_history = []

        # for data, mask in train_loader:
        for data, mask, label, _ in tqdm(train_loader, desc='training progress', leave=False):
            data, mask, label = data.to(args.device), mask.to(args.device), label.to(args.device)

            # network prediction
            out_seg, out_cla = net.seg_cla_forward(data)

            # compute loss
            loss_seg = criterion_dice(out_seg, mask) + criterion_ce(out_seg, mask)
            loss_cla = criterion_ce(out_cla, label)
            loss_seg_history.append(loss_seg.cpu().data.numpy())
            loss_cla_history.append(loss_cla.cpu().data.numpy())

            # backward propagation and parameter update
            optim.zero_grad()
            loss_total = loss_seg + loss_cla
            loss_total.backward()
            optim.step()

        logging.info('epoch: %d/%d | train | dice loss: %.4f | cla loss: %.4f' % (
            epoch, args.epoch, float(np.mean(loss_seg_history)), float(np.mean(loss_cla_history))))

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
                out_seg, out_cla = net.seg_cla_forward(data)
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

        logging.info('epoch: %d/%d | val | DICE: %.4f | PA: %.4f | IOU: %.4f | ACC: %.4f' % (
            epoch, args.epoch, dice / dice_total, pa / pa_total, iou / iou_total, acc / acc_total))

        if dice > best_dice:
            best_dice = dice
            torch.save(net.state_dict(), './model/net_sup_cla.pth')
            logging.info('best model | epoch: %d | DICE: %.4f | PA: %.4f | IOU: %.4f' % (
                epoch, dice / dice_total, pa / pa_total, iou / iou_total))


if __name__ == '__main__':
    logging.basicConfig(filename="log/train_sup_cla.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')

    args = parse_arg()
    args.labeled_ratio = 1.0
    logging.info(args)

    train_supervised(args)
