from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
import logging
import math
import numpy as np
import torch

import torchvision
from torch.utils.data import BatchSampler
from torch.utils.data import DataLoader
from torch.utils.data import Sampler
from torch.utils.data import SequentialSampler

from .dataset import CoverStegoDataset, OnTheFly  # 导入自定义数据集
from .transform import *  # 导入数据变换
from .. import utils  # 导入工具模块

logger = logging.getLogger(__name__)  # 获取日志记录器


class TrainingSampler(Sampler):
    """训练采样器，用于生成无限序列的索引。"""

    def __init__(self, size, seed=None, shuffle=True):
        """初始化采样器。

        Args:
            size (int): 数据集大小。
            seed (int, optional): 随机种子。默认为None。
            shuffle (bool, optional): 是否打乱数据。默认为True。
        """
        self._size = size
        self._shuffle = shuffle
        self._seed = seed if seed is not None else utils.get_random_seed()

    def __iter__(self):
        """迭代器，生成无限序列的索引。"""
        yield from itertools.islice(self._infinite_indices(), 0, None, 1)

    def _infinite_indices(self):
        """生成无限序列的索引，可以选择是否打乱。"""
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)


class BalancedBatchSampler(BatchSampler):
    """平衡批次采样器，确保每个批次中包含来自不同组的样本。"""

    def __init__(self, sampler, group_ids, batch_size):
        """初始化平衡批次采样器。

        Args:
            sampler (Sampler): 基础采样器。
            group_ids (list[int]): 每个样本的组ID。
            batch_size (int): 批次大小。
        """
        if not isinstance(sampler, Sampler):
            raise ValueError("sampler should be an instance of torch.utils.data.Sampler")

        self._sampler = sampler
        self._group_ids = np.asarray(group_ids)
        self._batch_size = batch_size
        groups = np.unique(self._group_ids).tolist()
        assert batch_size % len(groups) == 0, "批次大小必须能被组数整除"

        self._buffer_per_group = {k: [] for k in groups}
        self._group_size = batch_size // len(groups)

    def __iter__(self):
        """迭代器，生成平衡的批次索引。"""
        for idx in self._sampler:
            group_id = self._group_ids[idx]
            self._buffer_per_group[group_id].append(idx)
            if all(len(v) >= self._group_size for v in self._buffer_per_group.values()):
                idxs = []
                for v in self._buffer_per_group.values():
                    idxs.extend(v[:self._group_size])
                    del v[:self._group_size]
                yield np.random.permutation(idxs)

    def __len__(self):
        """返回迭代器的长度，此处不定义因为长度不固定。"""
        raise NotImplementedError("len() of GroupedBatchSampler is not well-defined.")


def build_train_loader(cover_dir, stego_dir, batch_size=32, num_workers=0):
    """构建训练数据加载器。

    Args:
        cover_dir (str): 覆盖图像目录。
        stego_dir (str): 隐写图像目录。
        batch_size (int, optional): 批次大小。默认为32。
        num_workers (int, optional): 工作线程数。默认为0。

    Returns:
        DataLoader: 训练数据加载器。
        int: 每个epoch的长度。
    """
    transform = torchvision.transforms.Compose([
        RandomRot(),  # 随机旋转
        RandomFlip(),  # 随机翻转
        ToTensor(),  # 转换为张量
    ])
    dataset = CoverStegoDataset(cover_dir, stego_dir, transform)

    size = len(dataset)
    sampler = TrainingSampler(size)
    batch_sampler = BalancedBatchSampler(sampler, dataset.labels, batch_size) if stego_dir else BatchSampler(sampler,
                                                                                                             batch_size,
                                                                                                             drop_last=False)
    epoch_length = math.ceil(size / batch_size)

    logger.info('Training set length is {}'.format(size))
    logger.info('Training epoch length is {}'.format(epoch_length))

    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_reset_seed,
    )
    return train_loader, epoch_length


def build_otf_train_loader(cover_dir, num_workers=0):
    """构建在线生成训练数据的数据加载器。

    Args:
        cover_dir (str): 覆盖图像目录。
        num_workers (int, optional): 工作线程数。默认为0。

    Returns:
        DataLoader: 训练数据加载器。
        int: 每个epoch的长度。
    """
    # 设置批量大小
    batch_size = 1

    # 定义数据预处理和增强的转换流程
    # RandomRot(): 随机旋转
    # RandomFlip(): 随机翻转
    # ToTensor(): 将图像转换为Tensor
    transform = torchvision.transforms.Compose([
        RandomRot(),
        RandomFlip(),
        ToTensor(),
    ])

    # 创建一个OnTheFly数据集实例，用于动态生成数据
    # cover_dir: 封面图片目录
    # transform: 应用到每个样本上的转换
    dataset = OnTheFly(cover_dir, transform=transform)

    # 获取数据集的大小（样本数量）
    size = len(dataset)

    # 创建一个训练采样器，用于随机采样样本
    sampler = TrainingSampler(size)

    # 创建一个批量采样器，用于将样本分组为批次
    # sampler: 采样器实例
    # batch_size: 每个批次的样本数量
    # drop_last: 如果最后一个批次不完整，是否丢弃
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)

    # 计算每个epoch的长度，即批次数
    # math.ceil: 向上取整，确保所有样本都被使用
    epoch_length = math.ceil(size / batch_size)

    # 记录训练集的长度和每个epoch的长度
    logger.info('Training set length is {}'.format(size))
    logger.info('Training epoch length is {}'.format(epoch_length))

    # 创建一个数据加载器，用于在训练过程中加载数据
    # dataset: 数据集实例
    # batch_sampler: 批量采样器
    # num_workers: 用于数据加载的子进程数量
    # worker_init_fn: 每个工作者进程启动时调用的函数，用于重置随机种子
    train_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        worker_init_fn=worker_init_reset_seed,
    )

    # 返回训练数据加载器和每个epoch的长度
    return train_loader, epoch_length


def build_val_loader(cover_dir, stego_dir, batch_size=32, num_workers=0):
    """构建验证数据加载器。

    Args:
        cover_dir (str): 覆盖图像目录。
        stego_dir (str): 隐写图像目录。
        batch_size (int, optional): 批次大小。默认为32。
        num_workers (int, optional): 工作线程数。默认为0。

    Returns:
        DataLoader: 验证数据加载器。
    """
    transform = torchvision.transforms.Compose([
        ToTensor(),
    ])
    dataset = CoverStegoDataset(cover_dir, stego_dir, transform)

    sampler = SequentialSampler(dataset)
    batch_sampler = BatchSampler(sampler, batch_size, drop_last=False)

    logger.info('Testing set length is {}'.format(len(dataset)))

    test_loader = DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
    )
    return test_loader


def worker_init_reset_seed(worker_id):
    """工作线程初始化函数，用于重置随机种子。

    Args:
        worker_id (int): 工作线程ID。
    """
    utils.set_random_seed(np.random.randint(2 ** 31) + worker_id)
