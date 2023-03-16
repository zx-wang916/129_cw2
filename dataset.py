import torch
import numpy as np

from PIL import Image
from torchvision.datasets import OxfordIIITPet, VisionDataset
from torchvision.transforms import transforms


IMG_SIZE = (256, 256)


class OxfordIIITPetSeg(VisionDataset):

    @classmethod
    def split_train_val(cls, root, train_ratio=0.9, labeled_ratio=0.2):
        # check if the dataset is downloaded
        _ = OxfordIIITPet(root, download=True)

        # load image file path
        data, mask = cls.load_data_path(root, True)

        # shuffle the dataset
        idx_shuffle = np.arange(len(data))
        np.random.shuffle(idx_shuffle)
        data = data[idx_shuffle]
        mask = mask[idx_shuffle]

        # split the data
        idx_train = int(len(data) * train_ratio)
        train_data = data[:idx_train]
        train_mask = mask[:idx_train]
        val_data = data[idx_train:]
        val_mask = mask[idx_train:]

        # initialize the train set
        train_set = OxfordIIITPetSeg(root, True, labeled_ratio)
        train_set.data = train_data
        train_set.mask = train_mask
        train_set.divide_dataset()

        # initialize the validation set
        val_set = OxfordIIITPetSeg(root, False)
        val_set.data = val_data
        val_set.mask = val_mask

        return train_set, val_set

    @classmethod
    def split_test(cls, root):
        # check if the dataset is downloaded
        _ = OxfordIIITPet(root, download=True)

        # load image file path
        data, mask = cls.load_data_path(root, True)

        # shuffle the dataset
        idx_shuffle = np.arange(len(data))
        np.random.shuffle(idx_shuffle)
        data = data[idx_shuffle]
        mask = mask[idx_shuffle]

        # initialize the test set
        test_set = OxfordIIITPetSeg(root, False)
        test_set.data = data
        test_set.mask = mask

        return test_set

    @staticmethod
    def load_data_path(root, train):
        data = []
        mask = []

        if train:
            file_name = 'trainval.txt'
        else:
            file_name = 'test.txt'

        with open(root + '/oxford-iiit-pet/annotations/' + file_name) as file:
            for line in file:
                image_filename, label, *_ = line.strip().split()
                mask.append(root + '/oxford-iiit-pet/annotations/trimaps/' + image_filename + '.png')
                data.append(root + '/oxford-iiit-pet/images/' + image_filename + '.jpg')

        data = np.array(data, dtype=object)
        mask = np.array(mask, dtype=object)
        return data, mask

    def __init__(self, root, train=True, labeled_ratio=0.2):
        super().__init__(root)
        self.train = train
        self.labeled_ratio = labeled_ratio

        self.data = None
        self.mask = None
        self.unlabeled = None

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

    def divide_dataset(self):
        # divide the dataset into labeled and unlabeled parts
        idx_labeled = int(len(self.data) * self.labeled_ratio)
        self.unlabeled = self.data[idx_labeled:]
        self.data = self.data[:idx_labeled]
        self.mask = self.mask[:idx_labeled]

    def __len__(self):
        if self.train:
            return len(self.data) + len(self.unlabeled)
        else:
            return len(self.data)

    def __getitem__(self, idx):
        if self.train:
            return self.getitem_train(idx)
        else:
            return self.getitem_test(idx)

    def getitem_train(self, idx):
        if idx < len(self.data):
            # sample labeled data with index
            image = Image.open(self.data[idx]).convert("RGB")
            mask = Image.open(self.mask[idx]).convert("L")

            # transform to Tensor before padding
            image = self.transform1(image)
            mask = self.transform1(mask)

            # pad the image to square image
            image = self.pad_to_square_image(image, 0)
            mask = self.pad_to_square_image(mask, 3)

            # resize the image and mask
            image = self.transform2(image)
            mask = self.transform3(mask)

            # make the mask into multichannel
            mask_out = torch.zeros((3, *IMG_SIZE))
            mask = mask.squeeze(0)
            mask_out[(0, *torch.where(mask == 1))] = 1
            mask_out[(1, *torch.where(mask == 2))] = 1
            mask_out[(2, *torch.where(mask == 3))] = 1

            return image, mask_out, 1

        else:
            # return unlabeled data
            idx -= len(self.data)

            # randomly sample unlabeled data
            unlabeled = Image.open(self.unlabeled[idx]).convert("RGB")
            unlabeled = self.transform1(unlabeled)
            unlabeled = self.pad_to_square_image(unlabeled, 0)
            unlabeled = self.transform2(unlabeled)

            return unlabeled, torch.empty_like(unlabeled), 0

    def getitem_test(self, idx):
        # sample labeled data with index
        image = Image.open(self.data[idx]).convert("RGB")
        mask = Image.open(self.mask[idx]).convert("L")

        # transform to Tensor before padding
        image = self.transform1(image)
        mask = self.transform1(mask)

        # pad the image to square image
        image = self.pad_to_square_image(image, 0)
        mask = self.pad_to_square_image(mask, 3)

        # resize the image and mask
        image = self.transform2(image)
        mask = self.transform3(mask)

        # make the mask into multichannel
        mask_out = torch.zeros((3, *IMG_SIZE))
        mask = mask.squeeze(0)
        mask_out[(0, *torch.where(mask == 1))] = 1
        mask_out[(1, *torch.where(mask == 2))] = 1
        mask_out[(2, *torch.where(mask == 3))] = 1

        return image, mask_out

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
