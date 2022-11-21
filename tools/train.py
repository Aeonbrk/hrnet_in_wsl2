# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from config import cfg
from config import update_config
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint
from utils.utils import create_logger
from utils.utils import get_model_summary

import dataset
import models


"""
1.解析参数
2.构建网络模型
3.加载训练测试数据集迭代器
4.迭代训练
5.模型评估保存
"""

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general,指定yaml文件的路径
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    # 暂时没有具体实现
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    # philly
    # 模型的目录
    parser.add_argument('--modelDir',
                        help='model directory',
                        type=str,
                        default='')

    # 输出log的目录
    parser.add_argument('--logDir',
                        help='log directory',
                        type=str,
                        default='')

    # 训练数据的目录
    parser.add_argument('--dataDir',
                        help='data directory',
                        type=str,
                        default='')

    # 预训练模型的目录
    parser.add_argument('--prevModelDir',
                        help='prev Model directory',
                        type=str,
                        default='')


    args = parser.parse_args()

    return args


def main():
    # 对输入参数进行解析
    args = parse_args()
    # 根据输入参数对cfg进行更新
    update_config(cfg, args)

    # 创建logger,用于记录训练过程的打印信息
    logger, final_output_dir, tb_log_dir = create_logger(
        cfg, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(cfg)


    # cudnn related setting
    # 使用GPU的一些相关设置
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # 根据配置文件构建网络
    print('models.'+cfg.MODEL.NAME+'.get_pose_net')
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=True
    )

    # copy model file,拷贝lib/models/pose_hrnet.py文件到输出目录之中
    this_dir = os.path.dirname(__file__)
    print(os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'))
    shutil.copy2(
        os.path.join(this_dir, '../lib/models', cfg.MODEL.NAME + '.py'),
        final_output_dir)
    # logger.info(pprint.pformat(model))

    # 用于训练信息的图形化显示
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    # 用于模型的图形化显示
    dump_input = torch.rand(
        (1, 3, cfg.MODEL.IMAGE_SIZE[1], cfg.MODEL.IMAGE_SIZE[0])
    )
    #writer_dict['writer'].add_graph(model, (dump_input, ))

    logger.info(get_model_summary(model, dump_input))



    # 让模型支持多GPU训练
    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()


    # define loss function (criterion) and optimizer,用于计算loss
    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()


    # Data loading code,对输入图象数据进行正则化处理
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # 创建训练以及测试数据的迭代器
    train_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=cfg.TRAIN.SHUFFLE,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=cfg.PIN_MEMORY
    )


    # 模型加载以及优化策略的相关配置
    best_perf = 0.0 #
    best_model = False
    last_epoch = -1
    optimizer = get_optimizer(cfg, model)
    begin_epoch = cfg.TRAIN.BEGIN_EPOCH
    checkpoint_file = os.path.join(
        final_output_dir, 'checkpoint.pth'
    )

    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info("=> loading checkpoint '{}'".format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)
        begin_epoch = checkpoint['epoch']
        best_perf = checkpoint['perf']
        last_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        logger.info("=> loaded checkpoint '{}' (epoch {})".format(
            checkpoint_file, checkpoint['epoch']))

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, cfg.TRAIN.LR_STEP, cfg.TRAIN.LR_FACTOR,
        last_epoch=last_epoch
    )

    #循环迭代进行训练
    for epoch in range(begin_epoch, cfg.TRAIN.END_EPOCH):
        lr_scheduler.step()

        # train for one epoch
        train(cfg, train_loader, model, criterion, optimizer, epoch,
              final_output_dir, tb_log_dir, writer_dict)


        # evaluate on validation set
        perf_indicator = validate(
            cfg, valid_loader, valid_dataset, model, criterion,
            final_output_dir, tb_log_dir, writer_dict
        )

        if perf_indicator >= best_perf:
            best_perf = perf_indicator
            best_model = True
        else:
            best_model = False

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        save_checkpoint({
            'epoch': epoch + 1,
            'model': cfg.MODEL.NAME,
            'state_dict': model.state_dict(),
            'best_state_dict': model.module.state_dict(),
            'perf': perf_indicator,
            'optimizer': optimizer.state_dict(),
        }, best_model, final_output_dir)

    final_model_state_file = os.path.join(
        final_output_dir, 'final_state.pth'
    )
    logger.info('=> saving final model state to {}'.format(
        final_model_state_file)
    )
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()

