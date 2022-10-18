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

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # 使用配置文件构建网络
    model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(cfg, is_train=False)
    # 'model.pose_hrnet.get_pose_net' ---> lib/models/pose_hrnet.py --->PoseHighResolutionNet
    # The source is a string representing a code object as returned by compile().
    # compile():将source编译为可以通过exec()或eval()执行的code对象。

    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)

    model = torch.nn.DataParallel(model, device_ids=cfg.GPUS).cuda()

    criterion = JointsMSELoss(
        use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT
    ).cuda()

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # 创建测试数据集及其迭代器
    valid_dataset = eval('dataset.'+cfg.DATASET.DATASET)(
        cfg, cfg.DATASET.ROOT, cfg.DATASET.TEST_SET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.TEST.BATCH_SIZE_PER_GPU*len(cfg.GPUS),
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True
    )

    # evaluate on validation set
    validate(cfg, valid_loader, valid_dataset, model, criterion,
             final_output_dir, tb_log_dir)
    # lib/core/function.py


if __name__ == '__main__':
    main()
