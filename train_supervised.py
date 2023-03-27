import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import get_sup_dataset
from model import ResUNet
from utils import create_dir, parse_arg
from utils import dice_loss, compute_metric
from tqdm import tqdm

create_dir()


def train_supervised(args):
    # prepare train and validation dataset
    train_set, val_set = get_sup_dataset('./data', args.train_val_ratio, args.labeled_ratio)

    # prepare dataloader
    train_loader = DataLoader(train_set, args.batch_size, True, num_workers=args.num_worker)
    val_loader = DataLoader(val_set, args.batch_size, True, num_workers=args.num_worker)

    # initialize network
    net = ResUNet()
    net = net.to(args.device)

    # define loss and optimizer
    # criterion = torch.nn.MSELoss()
    criterion = dice_loss
    optim = torch.optim.Adam(net.parameters(), lr=args.lr)

    print('start training!')
    for epoch in range(args.epoch):

        # ####################################### train model #######################################

        # training performance metrics
        pa = pa_total = 0
        iou = iou_total = 0
        dice = dice_total = 0
        loss_history = []

        # for data, mask in train_loader:
        for data, mask in tqdm(train_loader, desc='training progress', leave=False):
            data, mask = data.to(args.device), mask.to(args.device)

            # network predict
            out = net(data)

            # compute loss
            loss = criterion(out, mask)

            # backward propagation and parameter update
            optim.zero_grad()
            loss.backward()
            optim.step()

            loss_history.append(loss.cpu().data.numpy())

            out = torch.argmax(out, dim=1)
            result = compute_metric(out, mask)
            pa += result[0]
            pa_total += len(mask)
            iou += result[1]
            iou_total += len(mask)
            dice += result[2]
            dice_total += len(mask)

        print('epoch: %d/%d | train | DICE: %.3f | PA: %.3f | IOU: %.3f | loss: %.3f' % (
            epoch, args.epoch, dice / dice_total, pa / pa_total, iou / iou_total, float(np.mean(loss_history))))

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

                result = compute_metric(out, mask)
                pa += result[0]
                pa_total += len(mask)
                iou += result[1]
                iou_total += len(mask)
                dice += result[2]
                dice_total += len(mask)

        print('epoch: %d/%d |  val  | DICE: %.3f | PA: %.3f | IOU: %.3f' % (
            epoch, args.epoch, dice / dice_total, pa / pa_total, iou / iou_total))


if __name__ == '__main__':
    args = parse_arg()
    train_supervised(args)

    # import matplotlib.pyplot as plt
    # import matplotlib
    #
    # matplotlib.use('TkAgg')
    #
    # for i in torch.where(is_labeled == 1)[0]:
    #     if is_labeled[i] == 1:
    #         _, ax = plt.subplots(1, 2)
    #         ax[0].imshow(data[i].permute((1, 2, 0)))
    #         ax[1].imshow(mask[i].permute((1, 2, 0)))
    #         plt.show()
