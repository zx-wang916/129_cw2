import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir, parse_arg
from utils import dice_loss, compute_region, metric_dice, metric_iou, metric_pa
from tqdm import tqdm

create_dir()


def train_supervised(args):
    # prepare train and validation dataset
    train_set, val_set = OxfordIIITPetSeg.split_train_val('./data', args.train_val_ratio, args.labeled_ratio)

    # prepare dataloader
    train_loader = DataLoader(train_set, args.batch_size, True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, args.batch_size, True, num_workers=args.num_worker)

    # initialize network
    net = ResUNet()
    net = net.to(args.device)

    # define loss and optimizer
    criterion = dice_loss
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    print('start training!')
    for epoch in range(args.epoch):

        # ####################################### train model #######################################
        loss_history = []

        # for data, mask in train_loader:
        for data, mask, is_labeled in tqdm(train_loader, desc='training progress', leave=False):
            data, mask = data.to(args.device), mask.to(args.device)

            data_labeled = data[torch.where(is_labeled == 1)]
            mask_labeled = mask[torch.where(is_labeled == 1)]

            # network predict
            out = net(data_labeled)

            # compute loss
            loss = criterion(out, mask_labeled)

            # backward propagation and parameter update
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_history.append(loss.cpu().data.numpy())

        print('epoch: %d | train | dice loss: %.3f' % (epoch, float(np.mean(loss_history))))

        if epoch > 100:
            torch.save(net.state_dict(), './model/supervised/net_%d.pth' % epoch)

        # ####################################### validate model #######################################

        # performance metrics
        pa = pa_total = 0
        iou = iou_total = 0
        dice = dice_total = 0

        with torch.no_grad():
            for data, mask in tqdm(val_loader, desc='validation progress', leave=False):
                data, mask = data.to(args.device), mask.to(args.device)

                # network predict
                out = net(data)
                out = torch.argmax(out, dim=1)

                for i in range(3):
                    # compute binary mask for segmentation of each class
                    out_class_i = torch.zeros_like(out)
                    out_class_i[torch.where(out == i)] = 1
                    mask_class_i = mask[:, i]

                    region = compute_region(out_class_i, mask_class_i)

                    pa += torch.sum(metric_pa(*region))
                    pa_total += len(mask)

                    iou += torch.sum(metric_iou(*region))
                    iou_total += len(mask)

                    dice += torch.sum(metric_dice(*region))
                    dice_total += len(mask)

        print('epoch: %d | val | DICE: %.3f | PA: %.3f | IOU: %.3f' % (
            epoch, dice / dice_total, pa / pa_total, iou / iou_total))


if __name__ == '__main__':
    args = parse_arg()
    train_supervised(args)
