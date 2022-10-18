# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # Object for parsing command line strings into Python objects.
    # description -- A description of what the program does

    # general
    # 加载指定的yaml配置文件
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # python tools/test.py \
    #     --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml \
    #     TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth \
    #     TEST.USE_GT_BBOX False

    # 从这个调用的命令可以看出上面这个函数是干嘛的

    # Generally, these calls tell the ArgumentParser how to take the strings on the command line and turn them into objects.
    # https://docs.python.org/3/library/argparse.html#the-add-argument-method

    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # 模型目录
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')

    # 日志目录
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')

    # 训练数据目录
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')

    # 预训练模型目录
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')

    args = parser.parse_args()
    return args


def main():
    # 解析输入参数，并更新cfg
    args = parse_args()
    update_config(cfg, args)  # lib/config/default.py

    # 配置日志记录
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'valid')  # lib/utils/utils.py

    logger.info(pprint.pformat(args))
    logger.info(cfg)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # 使用配置文件构建网络
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    # 'model.pose_hrnet.get_pose_net' ---> lib/models/pose_hrnet.py --->PoseHighResolutionNet
    # The source is a string representing a code object as returned by compile().
    # compile():将source编译为可以通过exec()或eval()执行的code对象。

    if cfg.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
        # torch.nn.Module.load_state_dict()
        # Copies parameters and buffers from state_dict into this module and its descendants. If strict is True,
        # then the keys of state_dict must exactly match the keys returned by this module’s state_dict() function.
        # https://pytorch.org/docs/stable/generated/torch.load.html
        # TEST.MODEL_FILE models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth

    else:
        model_state_file = os.path.join(
            final_output_dir, 'final_state.pth'
        )
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    # 启用GPU的并行训练模型
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()
    # module (Module) – module to be parallelized
    # device_ids (list of python:int or torch.device) – CUDA devices (default: all devices)
    # https://pytorch.org/docs/stable/generated/torch.nn.DataParallel.html

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()
    # lib/core/loss.py

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # Normalize a tensor image with mean and standard deviation. This transform does not support PIL Image.
    # Given mean: (mean[1],...,mean[n]) and std: (std[1],..,std[n]) for n channels,
    # this transform will normalize each channel of the input torch.*Tensor i.e.,
    # output[channel] = (input[channel] - mean[channel]) / std[channel]
    # ---
    # mean (sequence) – Sequence of means for each channel.
    # std (sequence) – Sequence of standard deviations for each channel.
    # inplace (bool,optional) – Bool to make this operation in-place.
    # ---
    # return Normalized Tensor image.
    # https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html

    # 创建测试数据集及其迭代器
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    # https://pytorch.org/vision/stable/generated/torchvision.transforms.Compose.html
    # dataset.coco
    # class COCODataset(JointsDataset) -->class JointsDataset(Dataset) -->def __init__(self, cfg, root, image_set, is_train, transform=None)
    # torchvision.transforms.Compose(transforms)--->Composes several transforms together--->return transform

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader

    # torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=None, sampler=None, batch_sampler=None,
    # num_workers=0, collate_fn=None, pin_memory=False, drop_last=False, timeout=0, worker_init_fn=None,
    # multiprocessing_context=None, generator=None, *, prefetch_factor=2, persistent_workers=False, pin_memory_device='')

    # dataset (Dataset) – dataset from which to load the data.
    # shuffle (bool, optional) – set to True to have the data reshuffled at every epoch (default: False).
    # num_workers (int, optional) – how many subprocesses to use for data loading. 0 means that the data will be loaded in the main process. (default: 0)
    # pin_memory (bool, optional) – If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them.

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    # lib/core/function.py


if __name__ == '__main__':
    main()
