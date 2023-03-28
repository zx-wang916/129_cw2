import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_sup_dataset
from model import ResUNet
from utils import create_dir, parse_arg
from utils import dice_loss, compute_metric

create_dir()


def train_supervised(args):
    # prepare train and validation dataset
    train_set, val_set = get_sup_dataset('./data', args.train_val_ratio, args.labeled_ratio)

    # prepare dataloader
    train_loader = DataLoader(train_set, args.batch_size, True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, args.batch_size, True, num_workers=args.num_worker)

    # initialize network
    net = ResUNet().to(args.device)

    # define loss and optimizer
    criterion = dice_loss
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    print('start training!')
    for epoch in range(args.epoch):

        # ####################################### train model #######################################

        loss_history = []

        # for data, mask in train_loader:
        for data, mask in tqdm(train_loader, desc='training progress', leave=False):
            data, mask = data.to(args.device), mask.to(args.device)

            # network predict
            out = net.forward(data)

            # compute loss
            loss = criterion(out, mask)

            # backward propagation and parameter update
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_history.append(loss.cpu().data.numpy())

        print('epoch: %d/%d | train | dice loss: %.3f' % (epoch, args.epoch, float(np.mean(loss_history))))

        torch.save(net.state_dict(), './model/supervised/net_%d.pth' % epoch)

        # ####################################### validate model #######################################

        # validation performance metrics
        pa = pa_total = 0
        iou = iou_total = 0
        dice = dice_total = 0

        with torch.no_grad():
            for data, mask in tqdm(val_loader, desc='validation progress', leave=False):
                data, mask = data.to(args.device), mask.to(args.device)

                # network predict
                out = net(data)
                out = torch.argmax(out, dim=1)

                # compute metrics
                result = compute_metric(out, mask)
                pa += result[0]
                iou += result[1]
                dice += result[2]
                pa_total += len(mask)
                iou_total += len(mask)
                dice_total += len(mask)

        print('epoch: %d/%d |  val  | DICE: %.3f | PA: %.3f | IOU: %.3f' % (
            epoch, args.epoch, dice / dice_total, pa / pa_total, iou / iou_total))


if __name__ == '__main__':
    args = parse_arg()
    train_supervised(args)
