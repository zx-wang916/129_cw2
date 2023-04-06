import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import get_test_dataset
from model import ResUNet
from utils import create_dir, parse_arg, compute_metric
from PIL import Image

create_dir()


def test(net_path, args):
    # preparing dataset
    test_set = get_test_dataset(args.data_path)
    test_loader = DataLoader(test_set, args.batch_size, True, num_workers=args.num_worker)

    # initialize network
    net = ResUNet().to(args.device)
    net.load_state_dict(torch.load(net_path, map_location=args.device))

    # performance metrics
    pa = pa_total = 0
    iou = iou_total = 0
    dice = dice_total = 0

    with torch.no_grad():
        for data, mask in tqdm(test_loader, desc='testing progress', leave=False):
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

    dice = dice / dice_total
    pa = pa / pa_total
    iou = iou / iou_total
    print('test | DICE: %.3f | PA: %.3f | IOU: %.3f' % (dice, pa, iou))

    return dice, pa, iou


def visualization(net_path, out_path, num_sample, args):
    device = torch.device(args.device)

    with torch.no_grad():
        # preparing dataset
        test_set = get_test_dataset(args.data_path)

        # sample data for visualization
        sample_idx = np.random.randint(0, len(test_set), num_sample)

        img_all = Image.new('RGB', (256 * num_sample, 256 * 3))

        tr = transforms.ToPILImage()

        # initialize network
        net = ResUNet().to(device)
        net.load_state_dict(torch.load(net_path, map_location=device))

        for i, idx in enumerate(sample_idx):
            data, mask = test_set[idx]

            # network predict
            out = net(data.unsqueeze(0).to(device))
            out = torch.argmax(out, dim=1).squeeze(0)
            mask = torch.argmax(mask, dim=0)

            img_all.paste(tr(data), (224 * i, 224 * 0))

            mask = mask / 3 * 255
            mask = mask.numpy().astype(np.uint8)
            mask = Image.fromarray(mask, mode='L')
            mask = mask.convert('RGB')
            img_all.paste(mask, (224 * i, 224 * 1))

            out = out / 3 * 255
            out = out.cpu().numpy().astype(np.uint8)
            out = Image.fromarray(out, mode='L')
            out = out.convert('RGB')
            img_all.paste(out, (224 * i, 224 * 2))

        img_all.save(out_path)


if __name__ == '__main__':
    args = parse_arg()

    # test supervised model with all label available
    test('model/net_sup_full.pth', args)

    # test supervised model with 10% label available
    test('model/net_sup.pth', args)

    # test semi-supervised model
    test('model/net_semi.pth', args)

    # test semi-supervised model with classification label
    test('model/net_cla.pth', args)

    visualization('model/net_sup_full.pth', 'log/sup_full.png', 8, args)
    visualization('model/net_sup.pth', 'log/sup.png', 8, args)
    visualization('model/net_semi.pth', 'log/sup_semi.png', 8, args)
    visualization('model/net_cla.pth', 'log/sup_cla.png', 8, args)
