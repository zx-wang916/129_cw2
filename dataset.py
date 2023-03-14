import torch
import numpy as np

from PIL import Image
from torchvision.datasets import OxfordIIITPet, VisionDataset
from torchvision.transforms import transforms


IMG_SIZE = (256, 256)


# class OxfordIIITPetSeg(OxfordIIITPet):
#     def __init__(self, root, split, download=True):
#         super().__init__(root, split, 'segmentation', download=download)
#
#         self.multi_channel = True if split == 'trainval' else False
#
#         self.transform1 = transforms.Compose([transforms.PILToTensor()])
#         self.transform2 = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(IMG_SIZE),
#             transforms.ToTensor()
#         ])
#
#         self.transform3 = transforms.Compose([
#             transforms.ToPILImage(),
#             transforms.Resize(IMG_SIZE, transforms.InterpolationMode.NEAREST),
#             transforms.PILToTensor()
#         ])
#
#     def __getitem__(self, item):
#         data, mask = super(OxfordIIITPetSeg, self).__getitem__(item)
#
#         # transform to Tensor before padding
#         data = self.transform1(data)
#         mask = self.transform1(mask)
#
#         # pad the data and mask to a square image
#         height, width = data.shape[1:]
#         diff = abs(height - width)
#
#         if height > width:
#             pad = (diff // 2, diff - diff // 2, 0, 0)
#         else:
#             pad = (0, 0, diff // 2, diff - diff // 2)
#
#         data = torch.nn.functional.pad(data, pad, mode='constant', value=0)
#         mask = torch.nn.functional.pad(mask, pad, mode='constant', value=3)
#
#         # resize the image and mask, and remap the label from 123 to 012
#         data = self.transform2(data)
#         mask = self.transform3(mask) - 1
#
#         if self.multi_channel:
#             # make the mask into multichannel
#             mask_out = torch.zeros((3, *IMG_SIZE))
#             mask = mask.squeeze(0)
#             mask_out[(0, *torch.where(mask == 0))] = 1
#             mask_out[(1, *torch.where(mask == 1))] = 1
#             mask_out[(2, *torch.where(mask == 2))] = 1
#         else:
#             mask_out = mask.to(torch.float32)
#
#         return data, mask_out
#

class OxfordIIITPetSeg1(VisionDataset):
    def __init__(self, root, train=True, labeled_ratio=0.5):
        super().__init__(root)

        # download the dataset
        _ = OxfordIIITPet(root, download=True)

        self.train = train

        self.data = None
        self.mask = None
        self.unlabeled = None

        self.load_data_path(root, train)
        self.divide_dataset(labeled_ratio)

        self.transform1 = transforms.Compose([transforms.PILToTensor()])
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE, transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor()
        ])

        self.transform3 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE, transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor()
        ])

    def load_data_path(self, root, train):
        self.data = []
        self.mask = []

        if train:
            file_name = 'trainval.txt'
        else:
            file_name = 'test.txt'

        with open(root + '/oxford-iiit-pet/annotations/' + file_name) as file:
            for line in file:
                image_filename, label, *_ = line.strip().split()
                self.mask.append(root + '/oxford-iiit-pet/annotations/trimaps/' + image_filename + '.png')
                self.data.append(root + '/oxford-iiit-pet/images/' + image_filename + '.jpg')

        self.data = np.array(self.data, dtype=object)
        self.mask = np.array(self.mask, dtype=object)

    def divide_dataset(self, labeled_ratio):

        # shuffle the dataset
        idx_shuffle = np.arange(len(self))
        np.random.shuffle(idx_shuffle)

        self.data = self.data[idx_shuffle]
        self.mask = self.mask[idx_shuffle]

        # divide the dataset
        idx_labeled = int(len(self) * labeled_ratio)
        self.unlabeled = self.data[idx_labeled:]
        self.data = self.data[:idx_labeled]
        self.mask = self.mask[:idx_labeled]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # sample labeled data with index
        data = Image.open(self.data[idx]).convert("RGB")
        mask = Image.open(self.mask[idx]).convert("L")

        # transform to Tensor before padding
        data = self.transform1(data)
        mask = self.transform1(mask)

        # pad the image to square image
        data = self.pad_to_square_image(data, 0)
        mask = self.pad_to_square_image(mask, 3)

        # resize the image and mask
        data = self.transform2(data)
        mask = self.transform3(mask)

        # make the mask into multichannel
        mask_out = torch.zeros((3, *IMG_SIZE))
        mask = mask.squeeze(0)
        mask_out[(0, *torch.where(mask == 1))] = 1
        mask_out[(1, *torch.where(mask == 2))] = 1
        mask_out[(2, *torch.where(mask == 3))] = 1

        if self.train:
            # randomly sample unlabeled data
            unlabeled_idx = np.random.randint(0, len(self.unlabeled))
            unlabeled = Image.open(self.unlabeled[unlabeled_idx]).convert("RGB")
            unlabeled = self.transform1(unlabeled)
            unlabeled = self.pad_to_square_image(unlabeled, 0)
            unlabeled = self.transform2(unlabeled)

            return data, mask_out, unlabeled

        return data, mask_out

    @staticmethod
    def pad_to_square_image(img, pad_value):

        # pad the data and mask to a square image
        height, width = img.shape[1:]
        diff = abs(height - width)

        if height > width:
            pad = (diff // 2, diff - diff // 2, 0, 0)
        else:
            pad = (0, 0, diff // 2, diff - diff // 2)

        return torch.nn.functional.pad(img, pad, mode='constant', value=pad_value)
