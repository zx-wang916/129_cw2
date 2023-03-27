import torch
import numpy as np
import matplotlib

from matplotlib import pyplot as plt
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import get_test_dataset, get_semi_dataset
from model import ResUNet
from utils import create_dir, compute_region, metric_dice, metric_iou, metric_pa

create_dir()


def test(net_path, batch_size=32, device='cpu'):
    # preparing dataset
    test_set = get_test_dataset('./data')
    test_loader = DataLoader(test_set, batch_size, True, num_workers=8)

    # initialize network
    net = ResUNet().to(device)
    net.load_state_dict(torch.load(net_path, map_location=device))

    # performance metrics
    pa = pa_total = 0
    iou = iou_total = 0
    dice = dice_total = 0

    with torch.no_grad():
        for data, mask in tqdm(test_loader, desc='testing progress', leave=False):
            data, mask = data.to(device), mask.to(device)

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

        print('test | DICE: %.3f | PA: %.3f | IOU: %.3f' % (
            dice / dice_total, pa / pa_total, iou / iou_total))


def visualization(net_path, num_sample=4, device=torch.device('cpu')):
    with torch.no_grad():
        # preparing dataset
        test_set = get_test_dataset('./data')

        # use subplots to present the image
        _, ax = plt.subplots(3, num_sample)

        # sample data for visualization
        sample_idx = np.random.randint(0, len(test_set), num_sample)

        for i, idx in enumerate(sample_idx):
            data, mask = test_set[idx]
            mask = torch.argmax(mask, dim=0)

            # initialize network
            net = ResUNet().to(device)
            net.load_state_dict(torch.load(net_path, map_location=device))
            # net.eval()

            # network predict
            out = net(data.unsqueeze(0).to(device))
            out = torch.argmax(out, dim=1).squeeze(0)

            ax[0][i].imshow(data.permute((1, 2, 0)))
            ax[0][i].set_axis_off()
            ax[1][i].imshow(mask, 'gray')
            ax[1][i].set_axis_off()
            ax[2][i].imshow(out.cpu(), 'gray')
            ax[2][i].set_axis_off()
        plt.show()


if __name__ == '__main__':
    # # test best supervised model
    # print('best supervised model')
    # test('./model/supervised/net_132.pth', 128, 'cuda:4')
    #
    # # test best semi-supervised model
    # print('best semi-supervised model')
    # test('./model/semi/net_189.pth', 128, 'cuda:4')

    matplotlib.use('TkAgg')
    # visualization('./model/supervised/net_185.pth', 4, torch.device('cuda'))
    visualization('./model/semi/net_277.pth', 4, torch.device('cuda'))
