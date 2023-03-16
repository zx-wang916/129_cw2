import torch
import numpy as np

from torch.utils.data import DataLoader
from dataset import OxfordIIITPetSeg
from model import ResUNet
from utils import create_dir
from tqdm import tqdm

create_dir()


def test():
    # hyper-parameters
    BATCH_SIZE = 32
    DEVICE = torch.device('cuda:2')
    PATH = '.'

    # preparing dataset
    test_set = OxfordIIITPetSeg(PATH + '/data', split='test', download=True)
    test_loader = DataLoader(test_set, BATCH_SIZE, True, num_workers=4)

    # initialize network
    net = ResUNet()
    net = net.to(DEVICE)
    net.load_state_dict(torch.load(PATH + '/model/supervised/net_10.pth', map_location=DEVICE))

    # performance metrics
    PA = PA_TOTAL = 0
    IOU = IOU_TOTAL = 0

    with torch.no_grad():
        for data, mask in tqdm(test_loader, desc='testing progress', leave=False):
            data, mask = data.to(DEVICE), mask.to(DEVICE)

            # network predict
            out = net(data)
            out = torch.argmax(out, dim=1, keepdim=True)

            # compute the pixel accuracy metric
            PA += torch.sum(out == mask)
            PA_TOTAL += np.cumprod(mask.shape)[-1]

            # compute the IOU metric
            for i in range(3):
                out_class_i = torch.zeros_like(mask)
                out_class_i[torch.where(out == i + 1)] = 1
                mask_class_i = torch.zeros_like(mask)
                mask_class_i[torch.where(mask == i + 1)] = 1

                region_intersection = torch.sum(out_class_i * mask_class_i > 0, dim=(1, 2, 3))
                region_union = torch.sum(out_class_i + mask_class_i > 0, dim=(1, 2, 3))

                IOU += torch.sum(region_intersection / region_union)
                IOU_TOTAL += len(mask)

    print('PA: %.3f | IOU: %.3f' % (PA / PA_TOTAL, IOU / IOU_TOTAL))


if __name__ == '__main__':
    test()