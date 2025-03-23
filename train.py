from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#python train.py --train-cover-dir train/cover --val-cover-dir val/cover --train-stego-dir train/stego --val-stego-dir val/stego --model kenet --ckpt-dir checkpoints --random-crop --random-crop-train

import argparse
import logging
import os
import os.path as osp
import shutil
import random
import time

import torch
import torch.nn as nn
from torch.optim.adamax import Adamax
from torch.optim.adadelta import Adadelta

import src
from src import utils
from src.data import build_train_loader
from src.data import build_val_loader
from src.data import build_otf_train_loader
from src.matlab import matlab_speedy
# Import the summary writer
from torch.utils.tensorboard import SummaryWriter# Create an instance of the object

log_dir = os.path.join(os.getcwd(), 'log')
# 检查日志目录是否存在，如果不存在则创建
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
    # 创建SummaryWriter实例，指定日志目录

logger = logging.getLogger(__name__)
# 设置日志级别
logger.setLevel(logging.INFO)

# 创建一个日志格式器
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

# 创建一个日志处理器，这里使用的是控制台处理器
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# 将处理器添加到日志记录器
logger.addHandler(console_handler)

# 现在可以使用日志记录器了
logger.info('This is an info message.')
logger.info('Command Line Arguments: {}')

def parse_args():
    # 创建一个解析器对象
    parser = argparse.ArgumentParser()

    # 添加训练集封面图片目录参数
    parser.add_argument(
        '--train-cover-dir', dest='train_cover_dir', type=str, required=True,
        help='Directory path to the training cover images'
    )
    # 添加验证集封面图片目录参数
    parser.add_argument(
        '--val-cover-dir', dest='val_cover_dir', type=str, required=True,
        help='Directory path to the validation cover images'
    )
    # 添加训练集隐写图片目录参数
    parser.add_argument(
        '--train-stego-dir', dest='train_stego_dir', type=str, required=True,
        help='Directory path to the training stego images'
    )
    # 添加验证集隐写图片目录参数
    parser.add_argument(
        '--val-stego-dir', dest='val_stego_dir', type=str, required=True,
        help='Directory path to the validation stego images'
    )

    # 添加训练轮数参数
    parser.add_argument('--epoch', dest='epoch', type=int, default=500,
                        help='Number of epochs to train for, default is 500')
    # 添加学习率参数
    parser.add_argument('--lr', dest='lr', type=float, default=1e-3,
                        help='Learning rate for the optimizer, default is 0.001')
    # 添加权重衰减参数
    parser.add_argument('--wd', dest='wd', type=float, default=1e-4,
                        help='Weight decay for the optimizer, default is 0.0001')
    # 添加优化器epsilon参数
    parser.add_argument('--eps', dest='eps', type=float, default=1e-8,
                        help='Epsilon for the optimizer, default is 1e-8')
    # 添加alpha参数
    parser.add_argument('--alpha', dest='alpha', type=float, default=0.1,
                        help='Alpha value for loss function, default is 0.1')
    # 添加margin参数
    parser.add_argument('--margin', dest='margin', type=float, default=1.00,
                        help='Margin value for loss function, default is 1.00')

    # 添加随机裁剪参数
    parser.add_argument('--random-crop', dest='random_crop', action='store_true',
                        help='Enable random cropping of images')

    # 添加训练时随机裁剪参数
    parser.add_argument('--random-crop-train', dest='random_crop_train', action='store_true',
                        help='Enable random cropping of images during training, retrain strategy of SID')

    # 添加批量大小参数
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=32,
                        help='Batch size for training, default is 32')
    # 添加数据加载器工作线程数参数
    parser.add_argument('--num-workers', dest='num_workers', type=int, default=0,
                        help='Number of worker threads for data loading, default is 0')

    # 添加模型名称参数
    parser.add_argument('--model', dest='model', type=str, default='kenet',
                        help='Model name to use for training, default is "kenet"')
    # 添加微调模型路径参数
    parser.add_argument('--finetune', dest='finetune', type=str, default=None,
                        help='Path to the model to finetune, default is None')

    # 添加GPU ID参数
    parser.add_argument('--gpu-id', dest='gpu_id', type=int, default=0,
                        help='GPU ID to use for training, default is 0')
    # 添加随机种子参数
    parser.add_argument('--seed', dest='seed', type=int, default=-1,
                        help='Random seed for reproducibility, default is -1 (no seed set)')
    # 添加日志间隔参数
    parser.add_argument('--log-interval', dest='log_interval', type=int, default=10,
                        help='Interval for logging training progress, default is 10 iterations')
    # 添加检查点目录参数
    parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, required=True,
                        help='Directory to save checkpoints, required')
    # 添加学习率策略参数
    parser.add_argument('--lr-strategy', dest='lr_str', type=int, default=2,
                        help='Learning rate scheduler strategy, 1: StepLR, 2: MultiStepLR, 3: ExponentialLR, 4: CosineAnnealingLR, 5: ReduceLROnPlateau, default is 2')

    # 解析命令行参数
    args = parser.parse_args()
    return args


def setup(args):
    # 设定日志目录为当前目录下的"log"文件夹


    args.cuda = args.gpu_id >= 0
    if args.gpu_id >= 0:
        torch.cuda.set_device(args.gpu_id)

    log_file = osp.join(args.ckpt_dir, 'log.txt')
    utils.configure_logging(file=log_file, root_handler_type=0)

    utils.set_random_seed(None if args.seed < 0 else args.seed)

    logger.info('Command Line Arguments: {}')


args = parse_args()
setup(args)

logger.info('Building data loader')
# 从命令行参数中获取是否进行随机裁剪训练的标志
random_crop_train = args.random_crop_train

# 如果启用随机裁剪训练
if random_crop_train:
    # 使用build_otf_train_loader函数构建在线随机裁剪的训练数据加载器
    # 只需要封面图片目录和工作者数量
    train_loader, epoch_length = build_otf_train_loader(
        args.train_cover_dir, num_workers=args.num_workers
    )
else:
    # 如果不启用随机裁剪训练
    # 使用build_train_loader函数构建常规的训练数据加载器
    # 需要封面图片目录、隐写图片目录、批量大小和工作者数量
    train_loader, epoch_length = build_train_loader(
        args.train_cover_dir, args.train_stego_dir, batch_size=args.batch_size,
        num_workers=args.num_workers
    )

# 使用build_val_loader函数构建验证数据加载器
# 需要验证集的封面图片目录、隐写图片目录、批量大小和工作者数量
val_loader = build_val_loader(
    args.val_cover_dir, args.val_stego_dir, batch_size=args.batch_size,
    num_workers=args.num_workers
)

# 创建训练数据加载器的迭代器，用于在训练循环中获取数据
train_loader_iter = iter(train_loader)


logger.info('Building model')
if args.model == 'kenet':
    net = src.models.KeNet()
elif args.model == 'sid':
    net = src.models.SID()
else:
    raise NotImplementedError
if args.finetune is not None:
    net.load_state_dict(torch.load(args.finetune)['state_dict'], strict=False)





criterion_1 = nn.CrossEntropyLoss()
criterion_2 = src.models.ContrastiveLoss(margin=args.margin)

if args.cuda:
    net.cuda()
    criterion_1.cuda()
    criterion_2.cuda()

optimizer = Adamax(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)
if args.model == 'sid':
    optimizer = Adadelta(net.parameters(), lr=args.lr, eps=args.eps, weight_decay=args.wd)

lr_str = args.lr_str
if lr_str == 1:
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
elif lr_str == 2:
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[300, 400],
                                                     gamma=0.1)  # milestones=[900,975]
elif lr_str == 3:
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
elif lr_str == 4:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
elif lr_str == 5:
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.3,
                                                           patience=10, verbose=True, min_lr=0,
                                                           eps=1e-08)
else:
    raise NotImplementedError('Unsupported learning rate strategy')

def preprocess_data(images, labels, random_crop):
    """
    预处理图像和标签数据，包括调整维度、随机裁剪和分配到CUDA设备。

    参数:
    images : torch.Tensor
        输入图像张量，形状为(NxCxHxW)或(1xNxCxHxW)。
    labels : torch.Tensor
        输入标签张量，形状与images相同。
    random_crop : bool
        是否进行随机裁剪。

    返回:
    tuple
        预处理后的输入图像列表和标签张量。
    """

    # 检查图像张量是否具有额外的维度，例如通过unsqueeze添加的批次维度
    if images.ndim == 5:  # 如果形状是1xNxCxHxW
        images = images.squeeze(0)  # 移除第一个维度，变为NxCxHxW
        labels = labels.squeeze(0)  # 同样移除标签的第一个维度

    # 获取图像的高度和宽度
    h, w = images.shape[-2:]

    # 如果启用随机裁剪
    if random_crop:
        # 随机选择裁剪的高度和宽度，范围从3/4到全尺寸
        ch = random.randint(h * 3 // 4, h)
        cw = random.randint(w * 3 // 4, w)

        # 随机选择裁剪的起始点
        h0 = random.randint(0, h - ch)
        w0 = random.randint(0, w - cw)
    else:
        # 如果不进行随机裁剪，则裁剪区域为整个图像
        ch, cw, h0, w0 = h, w, 0, 0

    # 根据模型类型进行不同的处理
    if args.model == 'kenet':
        # 确保裁剪宽度为偶数，kenet模型可能需要
        cw = cw & ~1
        # 将图像分为两部分，可能用于kenet模型的双分支结构
        inputs = [
            images[..., h0:h0 + ch, w0:w0 + cw // 2],
            images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw]
        ]
    elif args.model == 'sid':
        # 对于sid模型，直接使用裁剪后的图像
        inputs = [images[..., h0:h0 + ch, w0:w0 + cw]]

    # 如果使用CUDA，将输入和标签移动到GPU上
    if args.cuda:
        inputs = [x.cuda() for x in inputs]  # 将每个输入张量移动到CUDA设备
        labels = labels.cuda()  # 将标签张量移动到CUDA设备

    return inputs, labels

def train(epoch):
    # 设置网络为训练模式
    net.train()
    # 初始化运行损失和运行准确率为0
    running_loss, running_accuracy = 0., 0.

    # 遍历每个批次
    for batch_idx in range(epoch_length):
        # 获取下一个批次的数据
        data = next(train_loader_iter)
        # 预处理数据
        inputs, labels = preprocess_data(data['image'], data['label'], args.random_crop)

        # 清零优化器的梯度
        optimizer.zero_grad()
        # 根据模型类型计算输出和损失
        if args.model == 'kenet':
            outputs, feats_0, feats_1 = net(*inputs)
            loss = criterion_1(outputs, labels) + \
                   args.alpha * criterion_2(feats_0, feats_1, labels)
        elif args.model == 'sid':
            outputs = net(*inputs)
            loss = criterion_1(outputs, labels)

        accuracy = src.models.accuracy(outputs, labels).item()
        running_accuracy += accuracy
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        # 记录训练信息

        # 每(log_interval)个批次后记录日志
        if (batch_idx + 1) % args.log_interval == 0:
            # 计算平均运行损失和运行准确率
            running_accuracy /= args.log_interval
            running_loss /= args.log_interval

            # 记录训练信息
            logger.info(
                f'Train epoch: {epoch} [{batch_idx + 1}/{epoch_length}]\tAccuracy: {100 * running_accuracy:.2f}%\tLoss: {running_loss:.6f}'
            )

            # 保存检查点
            is_best = False
            save_checkpoint(
                {
                    'iteration': batch_idx + 1,
                    'state_dict': net.state_dict(),
                    'best_prec1': running_accuracy,
                    'optimizer': optimizer.state_dict(),
                },
                is_best,
                filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
                best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))

            # 重置运行损失和运行准确率
            running_loss = 0.
            running_accuracy = 0.
            # 重新设置网络为训练模式
            net.train()


def valid():
    # 将模型设置为评估模式，关闭Dropout和Batch Normalization等训练特有的层
    net.eval()

    # 初始化验证集的损失和准确率
    valid_loss = 0.
    valid_accuracy = 0.

    # 使用torch.no_grad()来告诉PyTorch我们不会调用.backward()，从而减少内存消耗
    with torch.no_grad():
        # 遍历验证集的数据加载器
        for data in val_loader:
            # 对输入图像和标签进行预处理
            inputs, labels = preprocess_data(data['image'], data['label'], False)
            print(inputs)
            print(labels)
            # 根据不同的模型结构进行前向传播和损失计算
            if args.model == 'kenet':
                # KENet模型可能返回多个输出，包括最终输出和中间特征
                outputs, feats_0, feats_1 = net(*inputs)

                # 计算KENet模型的损失，包括最终输出的损失和中间特征的损失
                valid_loss += criterion_1(outputs, labels).item() + \
                              args.alpha * criterion_2(feats_0, feats_1, labels)
            elif args.model == 'sid':
                # SID模型只返回最终输出
                outputs = net(*inputs)

                # 计算SID模型的损失
                valid_loss += criterion_1(outputs, labels).item()

            # 计算准确率
            valid_accuracy += src.models.accuracy(outputs, labels).item()

    # 计算验证集的平均损失和平均准确率
    valid_loss /= len(val_loader)
    valid_accuracy /= len(val_loader)

    # 使用日志记录器记录验证集的损失和准确率
    logger.info(f'Test set: Loss: {valid_loss:.4f}, Accuracy: {100 * valid_accuracy:.2f}%')

    # 返回验证集的损失和准确率
    return valid_loss, valid_accuracy


def save_checkpoint(state, is_best, filename, best_name):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, best_name)


_time = time.time()
best_accuracy = 0.
for e in range(1, args.epoch + 1):

    logger.info(f'Epoch: {e}')

    logger.info('Train')
    train(e)
    logger.info(f'Time: {time.time() - _time}')
    logger.info('Test')
    valid_loss, accuracy = valid()
    logger.info(f'Test set: Loss: {valid_loss:.4f}, Accuracy: {100 * accuracy:.2f}%')

    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(accuracy)
    else:
        scheduler.step()
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        is_best = True
    else:
        is_best = False
    logger.info(f'Best accuracy: {best_accuracy}')
    logger.info(f'Time: {time.time() - _time}')
    save_checkpoint(
        {
            'epoch': e,
            'state_dict': net.state_dict(),
            'best_prec1': accuracy,
            'optimizer': optimizer.state_dict(),
        },
        is_best,
        filename=os.path.join(args.ckpt_dir, 'checkpoint.pth.tar'),
        best_name=os.path.join(args.ckpt_dir, 'model_best.pth.tar'))

