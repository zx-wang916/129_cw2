import torch
import numpy as np
import random

from PIL import Image
from torchvision.datasets import OxfordIIITPet, VisionDataset
from torchvision.transforms import transforms


IMG_SIZE = (256, 256)
FILL_IMAGE = 0
FILL_MASK = 3


def get_sup_dataset(root, train_ratio=0.9, labeled_ratio=0.2):
    # load and shuffle the raw data
    data, mask, _ = _load_shuffle(root, True)

    # split the data
    train_data, train_mask, _, val_data, val_mask, _ = _split_train_val(data, mask, None, train_ratio)

    # initialize the train set
    train_set = OxfordIIITPetSeg(root, 1, labeled_ratio, data=train_data, mask=train_mask)

    # initialize the validation set
    val_set = OxfordIIITPetSeg(root, 3, data=val_data, mask=val_mask)

    return train_set, val_set


def get_semi_dataset(root, train_ratio=0.9, labeled_ratio=0.2):
    # load and shuffle the raw data
    data, mask, _ = _load_shuffle(root, True)

    # split the data
    train_data, train_mask, _, val_data, val_mask, _ = _split_train_val(data, mask, None, train_ratio)

    # initialize the train set
    train_set = OxfordIIITPetSeg(root, 2, labeled_ratio, data=train_data, mask=train_mask)

    # initialize the validation set
    val_set = OxfordIIITPetSeg(root, 3, data=val_data, mask=val_mask)

    return train_set, val_set


def get_test_dataset(root, cla=False):
    # load and shuffle the raw data
    data, mask, label = _load_shuffle(root, False)
    return OxfordIIITPetSeg(root, 3, data=data, mask=mask, label=label)


def get_seg_cla_dataset(root, train_ratio=0.9, labeled_ratio=0.2):
    # load and shuffle the raw data
    data, mask, label = _load_shuffle(root, True)

    # split the data
    train_data, train_mask, train_label, val_data, val_mask, val_label = \
        _split_train_val(data, mask, label, train_ratio)

    # initialize the train set
    train_set = OxfordIIITPetSeg(
        root, 4, labeled_ratio, data=train_data, mask=train_mask, label=train_label)

    # initialize the validation set
    val_set = OxfordIIITPetSeg(
        root, 4, labeled_ratio=1, data=val_data, mask=val_mask, label=val_label)

    return train_set, val_set


def _split_train_val(data, mask, label, train_ratio):
    # split the data
    idx_train = int(len(data) * train_ratio)
    train_data = data[:idx_train]
    train_mask = mask[:idx_train]
    val_data = data[idx_train:]
    val_mask = mask[idx_train:]

    if label is None:
        return train_data, train_mask, None, val_data, val_mask, None

    else:
        train_label = label[:idx_train]
        val_label = label[idx_train:]
        return train_data, train_mask, train_label, val_data, val_mask, val_label


def _load_shuffle(root, train):
    # check if the dataset is downloaded
    _ = OxfordIIITPet(root, download=True)

    # load image file path
    data, mask, label = _load_data_path(root, train)

    # shuffle the dataset
    idx_shuffle = np.arange(len(data))
    np.random.shuffle(idx_shuffle)
    data = data[idx_shuffle]
    mask = mask[idx_shuffle]
    label = label[idx_shuffle]
    return data, mask, label


def _load_data_path(root, train):
    data = []
    mask = []
    label = []

    if train:
        file_name = 'trainval.txt'
    else:
        file_name = 'test.txt'

    with open(root + '/oxford-iiit-pet/annotations/' + file_name) as file:
        for line in file:
            image_filename, lab, *_ = line.strip().split()
            mask.append(root + '/oxford-iiit-pet/annotations/trimaps/' + image_filename + '.png')
            data.append(root + '/oxford-iiit-pet/images/' + image_filename + '.jpg')
            label.append(int(lab) - 1)

    data = np.array(data, dtype=object)
    mask = np.array(mask, dtype=object)
    label = np.array(label, dtype=object)
    return data, mask, label


class OxfordIIITPetSeg(VisionDataset):
    def __init__(self, root, split=1, labeled_ratio=0.2, **kwargs):
        super().__init__(root)

        # 1: supervised, 2: semi-supervised, 3: test, 4: with classification
        self.split = split
        self.labeled_ratio = labeled_ratio

        self.data = kwargs['data']
        self.mask = kwargs['mask']
        self.data_unlabeled = None

        if split == 1 or split == 2 or split == 4:
            self.divide_dataset()

            if split == 4:
                self.label = kwargs['label']

        self.tr_pil_to_tensor = transforms.PILToTensor()
        self.tr_to_tensor = transforms.ToTensor()
        self.tr_to_pil = transforms.ToPILImage()
        self.tr_augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90, fill=FILL_MASK),
            transforms.RandomResizedCrop(IMG_SIZE)
        ])
        self.tr_resize = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(IMG_SIZE)
        ])

    def __len__(self):
        if self.split == 1 or self.split == 3:
            return len(self.data)
        else:
            return len(self.data) + len(self.data_unlabeled)

    def __getitem__(self, idx):
        if self.split == 1 or self.split == 3:
            return self.getitem_without_unlabeled(idx)
        else:
            return self.getitem_with_unlabeled(idx)

    def divide_dataset(self):
        # divide the dataset into labeled and unlabeled parts
        idx_labeled = int(len(self.data) * self.labeled_ratio)
        self.data_unlabeled = self.data[idx_labeled:]
        self.data = self.data[:idx_labeled]
        self.mask = self.mask[:idx_labeled]

    def getitem_with_unlabeled(self, idx):
        if idx < len(self.data):
            # sample labeled data with index
            image = Image.open(self.data[idx]).convert("RGB")
            mask = Image.open(self.mask[idx]).convert("L")

            if self.split == 4:
                return *self.transform_data(image, mask), self.label[idx], 1
            else:
                return *self.transform_data(image, mask), 1

        else:
            # sample unlabeled data
            unlabeled = Image.open(self.data_unlabeled[idx - len(self.data)]).convert("RGB")

            if self.split == 4:
                return *self.transform_data(unlabeled, None), self.label[idx], 0
            else:
                return *self.transform_data(unlabeled, None), 0

    def getitem_without_unlabeled(self, idx):
        # sample labeled data with index
        image = Image.open(self.data[idx]).convert("RGB")
        mask = Image.open(self.mask[idx]).convert("L")

        if self.split == 4:
            return *self.transform_data(image, mask), self.label[idx]
        else:
            return self.transform_data(image, mask)

    def transform_data(self, image, mask):
        # transform to Tensor before padding
        image = self.tr_pil_to_tensor(image)

        # pad the image to square image
        image = self.pad_to_square_image(image, FILL_IMAGE)

        # for unlabeled data
        if mask is None:
            # resize the image
            image = self.tr_augmentation(image)
            image = self.tr_to_tensor(image)
            return image, torch.empty_like(image)

        # for labeled data
        else:
            # pad the mask
            mask = self.tr_pil_to_tensor(mask)
            mask = self.pad_to_square_image(mask, FILL_MASK)

            if self.split == 3:
                # for test dataset, we don't do argumentation
                image = self.tr_resize(image)
                mask = self.tr_resize(mask)

            else:
                # resize the image and mask in the same way
                seed = random.randint(0, 114514)
                torch.manual_seed(seed)
                image = self.tr_augmentation(image)
                torch.manual_seed(seed)
                mask = self.tr_augmentation(mask)

            image = self.tr_to_tensor(image)
            mask = self.tr_pil_to_tensor(mask)

            # make the mask multichannel
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
