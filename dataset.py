import copy
import os
import random

import albumentations as A
from albumentations.pytorch.transforms import ToTensor as AlbumToTensor
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchvision
from PIL import Image, ImageEnhance, ImageFilter
from pl_bolts.transforms.self_supervised import Patchify
from sklearn import preprocessing
from torch.utils.data import DataLoader, Dataset, random_split, sampler
from torchvision import transforms


def _affine_grid(size):
    if len(size) == 3:
        size = (1,) + size
    N, C, H, W = size
    grid = torch.FloatTensor(N, H, W, 2)
    linear_points = torch.linspace(-1, 1, W)
    grid[:, :, :, 0] = torch.ger(torch.ones(H), linear_points).expand_as(grid[:, :, :, 0])
    linear_points = torch.linspace(-1, 1, H)
    grid[:, :, :, 1] = torch.ger(linear_points, torch.ones(W)).expand_as(grid[:, :, :, 1])
    return grid


def random_baseline_drift(x, grid, magnitude=0.3, p=0.5):
    """用于分类数据增强的随机基线抖动，x是tensor"""
    if random.random() < p:
        grid = grid.clone()
        x_offset = random.random() * 5
        x_max = random.random() * 10
        magnitude = random.random() * magnitude  # <0.5
        a = torch.linspace(0 + x_offset, x_max + x_offset, 64)
        wave = torch.sin(a) * magnitude
        grid[:, :, :, 1] += wave
        return torch.nn.functional.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)
    else:
        return x


def vjitter(x, grid, magnitude=0.01, p=0.1):
    """ECG random high-frequency noise, x is PIL Image"""

    if random.random() < p:
        if random.random() < 0.5:  # 锐化增强噪声
            magnitude = random.randint(10, 20) * magnitude
            x = transforms.ToTensor()(
                x.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
            )
            grid = grid.clone()
            noise = torch.rand(1, 1, x.size(-1)) * magnitude
            grid[:, :, :, 1] += noise
            return torch.nn.functional.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)
        else:  # 弱抖动噪声
            magnitude = random.randint(5, 10) * magnitude
            x = transforms.ToTensor()(x)
            grid = grid.clone()
            noise = torch.rand(1, 1, x.size(-1)) * magnitude
            grid[:, :, :, 1] += noise
            return torch.nn.functional.grid_sample(x.unsqueeze(0), grid, align_corners=False).squeeze(0)
    else:
        return transforms.ToTensor()(x)


class RandomImageFilter(object):
    def __init__(self, p=0.5):
        self.filters = [
            ImageFilter.GaussianBlur(radius=0.8),
            ImageFilter.EDGE_ENHANCE,
            ImageFilter.SHARPEN,
            ImageFilter.SMOOTH_MORE,
        ]

    def __call__(self, img):
        _filter = random.choice(self.filters)
        return img.filter(_filter)


def rand_crop_lead(x):
    base_point = random.choice([49, 111, 170, 230, 295, 358, 418, 482, 545, 609, 670, 734])
    return x.crop((0, base_point - 32, 64, base_point + 32))


def cut_lead(x):
    """裁剪和合并12个导联"""
    base_point = [49, 111, 170, 230, 295, 358, 418, 482, 545, 609, 670, 734]
    xs = [x[:, i - 32 : i + 32, :] for i in base_point]
    return torch.cat(xs, dim=0)


class PairDataset(Dataset):
    """成对切片数据集，使用数据增强生成两组图片，用于训练simclr等"""

    def __init__(self, path, x_tfm=0.25):
        self.fileList = pd.read_csv(path)
        self.x_tfm = x_tfm
        self.affine_grid = _affine_grid((1, 64, 64))
        self.tfms_x = transforms.Compose([transforms.ToTensor(),])
        self.tfms_y = transforms.Compose(
            [
                transforms.ColorJitter(0.5, 0.5, 0.5),
                transforms.RandomAffine(degrees=5, translate=(0.3, 0.3)),
                RandomImageFilter(),
                transforms.Lambda(lambda x: vjitter(x, self.affine_grid, magnitude=0.02, p=0.1)),
                transforms.Lambda(lambda x: random_baseline_drift(x, self.affine_grid, magnitude=0.3, p=1)),
                transforms.RandomErasing(p=0.5, scale=(0.05, 0.05)),
            ]
        )

    def __len__(self):
        return len(self.fileList)  # 数据集的长度

    def __getitem__(self, idx):  # 重写索引方法，传入idx，获得各自属性
        img = Image.open(self.fileList["PATH"][idx] + "0.jpg").convert("L")
        img = rand_crop_lead(img)
        if random.random() < self.x_tfm:
            x = self.tfms_y(img)
        else:
            x = self.tfms_x(img)
        y = self.tfms_y(img)
        return x, y


class MultiLabelDataset(Dataset):
    """多标签分类数据集"""

    def __init__(self, labels, img_dir, train=True, label_dict=None):

        self.csv_path = labels
        self.img_dir = img_dir  # 图片文件目录
        df = pd.read_csv(labels)  # 读取labels.csv
        self.train = train
        self.fnames = df["fname"]
        self.labels = df["tags"]
        self.affine_grid = _affine_grid((1, 768, 64))
        if label_dict == None:
            self.label_dict = {
                "WPW": 0,
                "LAFB": 1,
                "MI": 2,
                "RBBB": 3,
                "LBBB": 4,
                "IAVB": 5,
                "RAE": 6,
                "LVH": 7,
                "RVH": 8,
            }
        else:
            self.label_dict = label_dict

        self.tfms_train = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: cut_lead(x)),])
        self.tfms_test = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: cut_lead(x)),])

        # 生成多标签targets
        self.targets = torch.zeros(len(self.fnames), len(self.label_dict))
        for n, line in enumerate(self.labels):
            for label in line.split():
                if label in self.label_dict.keys():
                    self.targets[n, self.label_dict[label]] = 1

        # 样本计数，用于调整权重
        ctg_sum = self.targets.sum(0, keepdim=True)
        self.target_weights = (ctg_sum * self.targets).sum(1)
        other_num = len(self.target_weights) - ctg_sum.sum()
        self.target_weights[
            torch.nonzero(
                torch.where(self.target_weights == 0, torch.FloatTensor([1]), torch.FloatTensor([0])), as_tuple=False
            )
        ] = other_num
        self.target_weights = 1 / self.target_weights

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_dir = self.img_dir + fname + "/"
        if self.train:
            sub_imgs = [i for i in os.listdir(img_dir) if "jpg" in i]
            img_name = random.choice(sub_imgs)
        else:
            img_name = "0.jpg"  # 失去了随机化
        img_path = img_dir + img_name
        img = Image.open(img_path).convert("L")

        if self.train:
            x = self.tfms_train(img)
        else:
            x = self.tfms_test(img)

        y = self.targets[idx]
        return x, y


class ECGSSLDataModule(pl.LightningDataModule):
    """训练moco，simclr等"""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_ds = PairDataset("label/PCUT260K_trainList.csv")
        self.valid_ds = PairDataset("label/PCUT260K_testList.csv")
        self.num_samples = len(self.train_ds)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.num_workers,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=self.hparams.num_workers,
        )


class ECGCPCDataset(Dataset):
    """cpc训练数据集, 兼用于detself训练"""

    def __init__(self, path, patch_size=8):
        self.fileList = pd.read_csv(path)
        if patch_size:
            self.tfms = transforms.Compose([transforms.ToTensor(), Patchify(patch_size, patch_size // 2)])
        else:
            self.tfms = transforms.ToTensor()

    def __len__(self):
        return len(self.fileList)  # 数据集的长度

    def __getitem__(self, idx):  # 重写索引方法，传入idx，获得各自属性
        img = Image.open(self.fileList["PATH"][idx] + "0.jpg").convert("L")
        img = rand_crop_lead(img)
        x = self.tfms(img)
        return x


class ECGCPCDataModule(pl.LightningDataModule):
    """训练CPC"""

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_ds = ECGCPCDataset("label/PCUT260K_trainList.csv", patch_size=hparams.patch_size)
        self.valid_ds = ECGCPCDataset("label/PCUT260K_testList.csv", patch_size=hparams.patch_size)
        self.num_samples = len(self.train_ds)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.hparams.batch_size, shuffle=True, num_workers=20, drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds, batch_size=self.hparams.batch_size, shuffle=False, drop_last=True, num_workers=20
        )


class MultiLabelDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        label_dict = {"WPW": 0, "LAFB": 1, "MI": 2, "RBBB": 3, "LBBB": 4, "IAVB": 5, "RAE": 6, "LVH": 7}
        self.train_ds = MultiLabelDataset(
            labels=self.hparams.train_label, img_dir="data/EVD300K_crops/", label_dict=label_dict
        )
        self.valid_ds = MultiLabelDataset(
            "label/ECG200K_ctg8_valid.csv", img_dir="data/EVD300K_crops/", label_dict=label_dict, train=False
        )
        self.test_ds = MultiLabelDataset(
            "label/ECG200K_ctg8_test.csv", img_dir="data/EVD300K_crops/", label_dict=label_dict, train=False
        )

        self.num_samples = len(self.train_ds)

    def train_dataloader(self):
        if self.hparams.weight_sample:
            spl = sampler.WeightedRandomSampler(
                self.train_ds.target_weights, int(len(self.train_ds) / self.hparams.epoch_cut), replacement=True
            )
            return DataLoader(
                self.train_ds,
                batch_size=self.hparams.batch_size,
                sampler=spl,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            )
        else:
            return DataLoader(
                self.train_ds,
                batch_size=self.hparams.batch_size,
                shuffle=true,
                num_workers=self.hparams.num_workers,
                pin_memory=True,
            )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )


class DetDataset(Dataset):
    """用于训练det预训练模型，预测各种心电图机自动计算的指标"""

    def __init__(
        self, labels, img_dir, tags=[3, 4, 6, 8, 9], train=True, label_dict=None, train_aug=False, return_fname=False
    ):

        self.csv_path = labels
        self.img_dir = img_dir  # 图片文件目录
        df = pd.read_csv(labels)  # 读取labels.csv
        self.train = train
        self.fnames = df["fname"]
        self.labels = df["tags"]
        self.return_fname = return_fname

        self.det = df.iloc[:, 1:11].values.astype(np.float32)
        #  self.det = preprocessing.scale(self.det, axis=0)  # 归一化
        # 0sex,1age,2HR,3PR,4QRS,5QRS_axis,6QT,7QTc,8RV5,9SV1
        #  selected_idx = [3, 4, 6, 8, 9]
        #  selected_idx = [3, 4, 6]
        selected_idx = tags

        #  selected_idx = [6]
        self.det = self.det[:, selected_idx]  # 0PR,1QRS,2QT,3RV5,4SV1
        self.det[:, [0, 1, 2]] /= 100
        #  self.det[:,[0]]/=100

        if label_dict == None:
            self.label_dict = {
                "WPW": 0,
                "LAFB": 1,
                "MI": 2,
                "RBBB": 3,
                "LBBB": 4,
                "IAVB": 5,
                "RAE": 6,
                "LVH": 7,
                "RVH": 8,
            }
        else:
            self.label_dict = label_dict

        if train_aug:
            self.affine_grid = _affine_grid((1, 768, 64))
            self.tfms_train = transforms.Compose(
                [
                    RandomImageFilter(),
                    transforms.Lambda(lambda x: vjitter(x, self.affine_grid, magnitude=0.02, p=0.1)),
                    transforms.Lambda(lambda x: random_baseline_drift(x, self.affine_grid, magnitude=0.3, p=1)),
                    transforms.RandomErasing(p=0.8, scale=(0.1, 0.1)),
                    transforms.Lambda(lambda x: cut_lead(x)),
                ]
            )
        else:
            self.tfms_train = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: cut_lead(x)),])
        self.tfms_test = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: cut_lead(x)),])

        # 生成多标签targets
        self.targets = torch.zeros(len(self.fnames), len(self.label_dict))
        for n, line in enumerate(self.labels):
            for label in line.split():
                if label in self.label_dict.keys():
                    self.targets[n, self.label_dict[label]] = 1

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img_dir = self.img_dir + fname + "/"
        if self.train:
            sub_imgs = [i for i in os.listdir(img_dir) if "jpg" in i]
            img_name = random.choice(sub_imgs)
        else:
            img_name = "0.jpg"  # 失去了随机化
        img_path = img_dir + img_name
        img = Image.open(img_path).convert("L")

        if self.train:
            x = self.tfms_train(img)
        else:
            x = self.tfms_test(img)

        y = self.targets[idx]
        d = self.det[idx]
        if self.return_fname:
            return x, d, fname
        else:
            return x, d


class DetDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        label_dict = {"WPW": 0, "LAFB": 1, "MI": 2, "RBBB": 3, "LBBB": 4, "IAVB": 5, "RAE": 6, "LVH": 7}
        train_aug = hparams.train_aug
        # TODO新的拆分模式，使用ECG260all
        self.train_ds = DetDataset(
            labels="label/ECG200K_ctg8_train.csv",
            img_dir="data/EVD300K_crops/",
            label_dict=label_dict,
            train_aug=train_aug,
            tags=hparams.tags,
        )
        self.valid_ds = DetDataset(
            labels="label/ECG200K_ctg8_valid.csv",
            img_dir="data/EVD300K_crops/",
            label_dict=label_dict,
            train=False,
            tags=hparams.tags,
        )
        self.test_ds = DetDataset(
            labels="label/ECG200K_ctg8_test.csv",
            img_dir="data/EVD300K_crops/",
            label_dict=label_dict,
            train=False,
            tags=hparams.tags,
        )
        self.num_samples = len(self.train_ds)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )

    def test_dataloade(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.hparams.num_workers,
        )


def batch2pil(x, nrow=8, normalize=True, padding=1, pad_value=1, range=None):
    grid = torchvision.utils.make_grid(
        x, normalize=normalize, nrow=nrow, pad_value=pad_value, padding=padding, range=range
    )
    return torchvision.transforms.ToPILImage()(grid.cpu())


if __name__ == "__main__":
    pass
