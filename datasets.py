import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import skimage.io as io
import glob
import numpy as np
import random


def get_images(filename):
    image_names = np.genfromtxt(filename, dtype=str)
    return image_names


def load_image(file):
    return Image.open(file)


class coseg_train_dataset(Dataset):
    def __init__(self, data_dir, label_dir, traintxt, input_transform=None, label_transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.traintxt = traintxt
        self.train_names = get_images(self.traintxt)

    def __getitem__(self, index):

        imagename1 = self.data_dir + self.train_names[index][0] + ".jpg"
        imagename2 = self.data_dir + self.train_names[index][1] + ".jpg"
        labelname1 = self.label_dir + self.train_names[index][2] + ".png"
        labelname2 = self.label_dir + self.train_names[index][3] + ".png"

        with open(imagename1, "rb") as f:
            image1 = load_image(f).convert('RGB')
        with open(imagename2, "rb") as f:
            image2 = load_image(f).convert('RGB')

        with open(labelname1, "rb") as f:
            label1 = load_image(f).convert('L')
        with open(labelname2, "rb") as f:
            label2 = load_image(f).convert('L')

        # random horizontal flip
        if random.random() < 0.5:
            image1 = image1.transpose(Image.FLIP_LEFT_RIGHT)
            label1 = label1.transpose(Image.FLIP_LEFT_RIGHT)

        # random horizontal flip
        if random.random() < 0.5:
            image2 = image2.transpose(Image.FLIP_LEFT_RIGHT)
            label2 = label2.transpose(Image.FLIP_LEFT_RIGHT)

        if self.input_transform is not None:
            image1 = self.input_transform(image1)
            image2 = self.input_transform(image2)

        if self.label_transform is not None:
            label1 = self.label_transform(label1)
            label2 = self.label_transform(label2)

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.train_names)


class coseg_val_dataset(Dataset):
    def __init__(self, data_dir, label_dir, val_txt, input_transform=None, label_transform=None):
        self.data_dir = data_dir
        self.label_dir = label_dir
        self.input_transform = input_transform
        self.label_transform = label_transform
        self.val_txt = val_txt
        self.val_names = get_images(self.val_txt)

    def __getitem__(self, index):

        imagename1 = self.data_dir + self.val_names[index][0] + ".jpg"
        imagename2 = self.data_dir + self.val_names[index][1] + ".jpg"
        labelname1 = self.label_dir + self.val_names[index][2] + ".png"
        labelname2 = self.label_dir + self.val_names[index][3] + ".png"

        with open(imagename1, "rb") as f:
            image1 = load_image(f).convert('RGB')
        with open(imagename2, "rb") as f:
            image2 = load_image(f).convert('RGB')

        with open(labelname1, "rb") as f:
            label1 = load_image(f).convert('L')
        with open(labelname2, "rb") as f:
            label2 = load_image(f).convert('L')

        if self.input_transform is not None:
            image1 = self.input_transform(image1)
            image2 = self.input_transform(image2)

        if self.label_transform is not None:
            label1 = self.label_transform(label1)
            label2 = self.label_transform(label2)

        return image1, image2, label1, label2

    def __len__(self):
        return len(self.val_names)
