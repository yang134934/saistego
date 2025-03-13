from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import numpy as np
from torch.utils.data import Dataset
from PIL import Image
import cv2

from ..matlab import S_UNIWARD  # 导入S_UNIWARD隐写算法


class CoverStegoDataset(Dataset):
    """封面和隐写数据集类，用于加载封面和隐写图像。"""

    def __init__(self, cover_dir, stego_dir, transform=None):
        """
        初始化函数。

        Args:
            cover_dir (str): 封面图像目录路径。
            stego_dir (str): 隐写图像目录路径，如果为None则只处理封面图像。
            transform (callable, optional): 应用于样本的变换函数。
        """
        self._transform = transform

        # 获取图像路径和标签
        self.images, self.labels = self.get_items(cover_dir, stego_dir)

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.images)

    def __getitem__(self, idx):
        """根据索引获取样本。

        Args:
            idx (int): 样本索引。

        Returns:
            dict: 包含图像和标签的样本字典。
        """
        image = np.array(Image.open(self.images[idx]))
        image = np.expand_dims(image, 2)  # 调整图像数组为(H, W, C)
        assert image.ndim == 3  # 确保图像是三维数组

        sample = {
            'image': image,
            'label': self.labels[idx]
        }

        if self._transform:
            sample = self._transform(sample)
        return sample

    @staticmethod
    def get_items(cover_dir, stego_dir):
        """静态方法，用于获取图像路径和标签列表。

        Args:
            cover_dir (str): 封面图像目录路径。
            stego_dir (str): 隐写图像目录路径。

        Returns:
            tuple: 包含图像路径列表和标签列表的元组。
        """
        images, labels = [], []

        cover_names = sorted(os.listdir(cover_dir))
        if stego_dir is not None:
            stego_names = sorted(os.listdir(stego_dir))
            assert cover_names == stego_names  # 确保封面和隐写图像文件名一致

        file_names = cover_names
        if stego_dir is None:
            dir_to_label = [(cover_dir, 0), ]  # 只处理封面图像
        else:
            dir_to_label = [(cover_dir, 0), (stego_dir, 1)]  # 处理封面和隐写图像

        for image_dir, label in dir_to_label:
            for file_name in file_names:
                image_path = osp.join(image_dir, file_name)
                if not osp.isfile(image_path):
                    raise FileNotFoundError('{} not exists'.format(image_path))
                images.append(image_path)
                labels.append(label)

        return images, labels


class OnTheFly(Dataset):
    """动态生成隐写图像的数据集类。"""

    def __init__(self, cover_dir, num=16, payload=0.4, transform=None):
        """
        初始化函数。

        Args:
            cover_dir (str): 封面图像目录路径。
            num (int, optional): 每个封面图像生成的样本数量。默认为16。
            payload (float, optional): 隐写载荷，即嵌入率。默认为0.4。
            transform (callable, optional): 应用于样本的变换函数。
        """
        self._transform = transform
        self._num = num
        self._payload = payload

        self.cover_label = 0
        self.stego_label = 1

        # 获取封面图像路径
        self.images = self.get_items(cover_dir)

    def __len__(self):
        """返回数据集中的样本数量。"""
        return len(self.images)

    def __getitem__(self, item):
        """根据索引动态生成隐写图像样本。

        Args:
            item (int): 样本索引。

        Returns:
            dict: 包含图像和标签的样本字典。
        """
        image = self.images[item]
        image = cv2.imread(image, flags=cv2.IMREAD_GRAYSCALE).astype(np.float64)  # 读取图像为灰度图

        h, w = image.shape
        # 随机裁剪尺寸
        crop_h = np.random.randint(h * 3 // 4, h + 1)
        crop_w = np.random.randint(w * 3 // 4, w + 1)

        # 随机裁剪位置
        h0s = np.random.randint(0, h - crop_h + 1, (self._num,))
        w0s = np.random.randint(0, w - crop_w + 1, (self._num,))

        new_images, new_labels = [], []
        for h0, w0 in zip(h0s, w0s):
            cover_img = image[h0: h0 + crop_h, w0: w0 + crop_w]
            new_images.append(cover_img)
            new_labels.append(self.cover_label)

            # 生成隐写图像
            stego_img = S_UNIWARD(cover_img, self._payload)
            new_images.append(stego_img)
            new_labels.append(self.stego_label)

        # 随机打乱顺序
        idxs = np.random.permutation(len(new_images))

        new_images = np.stack(new_images, axis=0)  # 调整数组为(N, H, W)
        new_images = new_images[idxs]
        new_images = new_images[:, :, :, None]  # 调整数组为(N, H, W, C)

        new_labels = np.asarray(new_labels)
        new_labels = new_labels[idxs]

        sample = {
            'image': new_images,
            'label': new_labels,
        }

        if self._transform:
            sample = self._transform(sample)

        return sample

    @staticmethod
    def get_items(cover_dir):
        """静态方法，用于获取封面图像路径列表。

        Args:
            cover_dir (str): 封面图像目录路径。

        Returns:
            list: 封面图像路径列表。
        """
        file_names = sorted(os.listdir(cover_dir))

        images = []
        for file_name in file_names:
            image_file = osp.join(cover_dir, file_name)
            if not osp.isfile(image_file):
                raise FileNotFoundError('{} not exists'.format(image_file))
            images.append(image_file)

        return images
