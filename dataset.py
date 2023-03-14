import torch

from torchvision.datasets import OxfordIIITPet
from torchvision.transforms import transforms


IMG_SIZE = (256, 256)


class OxfordIIITPetSeg(OxfordIIITPet):
    def __init__(
            self,
            root='./data',
            split="trainval",
            download=True):
        super().__init__(root, split, 'segmentation', download=download)

        self.transform1 = transforms.Compose([transforms.ToTensor()])
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE),
            transforms.ToTensor()
        ])

    def __getitem__(self, item):
        data, mask = super(OxfordIIITPetSeg, self).__getitem__(item)

        # transform to Tensor before padding
        data = self.transform1(data)
        mask = self.transform1(mask)

        # pad the data and mask to a square image
        height, width = data.shape[1:]
        diff = abs(height - width)

        if height > width:
            pad = (diff // 2, diff - diff // 2, 0, 0)
        else:
            pad = (0, 0, diff // 2, diff - diff // 2)

        data = torch.nn.functional.pad(data, pad, mode='constant', value=0)
        mask = torch.nn.functional.pad(mask, pad, mode='constant', value=2 / 255)

        # resize the image and mask
        data = self.transform2(data)
        mask = self.transform2(mask) * 255

        print(torch.unique(mask))

        return data, mask
