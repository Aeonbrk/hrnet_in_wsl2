# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn


BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


# 1.基础框架：
# 1）resnet网络的两种形式：class BasicBlock(nn.Module)
#                        class Bottleneck(nn.Module)
# 2）关键部分：高分辨率模块：class HighResolutionModule(nn.Module)
#             该模块的重要函数：def _make_one_branch
#                             def _make_branches
#                             def _make_fuse_layers
# 2.关键点预测模块【完整的网络】： class PoseHighResolutionNet(nn.Module)
#             该模块的重要函数：def _make_transition_layer
#                             def _make_layer
#                             def _make_stage


def conv3x3(in_planes, out_planes, stride=1):
    # 返回一个核为3×3，步长为1，padding为1的2d卷积层
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    # expansion是对输出通道数的倍乘，在basic中expansion是1，此时完全忽略expansion，输出的通道数就是plane，
    # 然而bottleneck就是不走寻常路，它的任务就是要对通道数进行压缩，再放大，于是，plane不再代表输出的通道数，而是block内部压缩后的通道数，输出通道数变为plane*expansion。

    # 初始化 layer 结构，赋值参数和 models
    # inplane是输入的通道数，plane是输出的通道数
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        # (1) 每个卷积块后面连接BN层进行归一化，并使用ReLU激活函数，将特征图映射到新的特征图上
        # 由于激活函数在 0 附近变化率大，因此被映射到新的特征空间的特征图也更容易被区别开来，这样既帮助了分类，也同时缓解了梯度消失问题。

        out = self.conv2(out)
        out = self.bn2(out)

        # ResNet18 中，每经过两个 BasicBlock，output 的尺寸（维度）会降低为原来的一半（例如从
        # 64*56*56 到 128*28*28）。而如果如果上一个 BasicBlock 的输出维度和当前的 BasicBlock 的
        # 维度不一样，那么上一个 BasicBlock 的输出 x（同时也是当前 BasicBlock 的输入）是不能和
        # 当前计算出的 output 直接相加的，这时就需要对 x 进行 downsample 处理。如果维度一致，
        # 例如 layer1 的 output 必然和最初的 input 一致；或者在后面的 layer 中，第二个 BasicBlock
        # 维度必然与上一个 BasicBlock 一致，这种情况下直接相加就行了，out+=x。
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        # 1x1,3x3,1x1,分别用来压缩维度，卷积处理，恢复维度
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        """
           :param num_branches: 当前stage分支的平行子网络数目
           :param blocks: BasicBlock或者Bottleneck
           :param num_blocks: BasicBlock或者Bottleneck数目

           :param num_inchannels: 输入通道数
               stage=2时, num_inchannels=[256]
               stage=3时, num_inchannels=[32,64]
               stage=4时, num_inchannels=[32,64,128]

           :param num_channels: 输出通道数
               stage=2时, num_channels=[32,64]
               stage=3时, num_channels=[32,64,128]
               stage=4时, num_channels=[32,64,128,256]

           :param fuse_method: 融合方式,默认为fuse

           :param multi_scale_output: HRNet-W32配置下,所有stage均为multi_scale_output=True
        """

        # 显式初始化 HighResolutionModule
        super(HighResolutionModule, self).__init__()

        # 检测输入参数
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        # 为每个分支构建分支网络
        # stage=2,3,4，平行子网分支数num_branches=2,3,4
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)

        # 创建多尺度融合层
        # stage=2,3,4时，len(self.fuse_layers)分别为2,3,4，与其num_branches在每个stage的数目是一致的
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):

        # 对输入的一些参数进行检测，判断num_branches 和 num_blocks,
        # num_inchannels, num_channels (list) 三者的长度是否一致，不一致则报错
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)  # error_message

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        # 该函数作用为搭建一个分支，由num_branches[branch_index]个BasicBlock或Bottleneck组成。
        downsample = None

        # 如果stride不为1, 或者输入通道数目与输出通道数目不一致，使用1×1卷积改变通道数
        if stride != 1 or \
           self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []

        # 为当前分支branch_index创建一个block,在该处进行下采样
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )

        # 将输出通道数赋值给输入通道数,为下一stage作准备
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion

        # 为[1, num_blocks[branch_index]]分支创建block
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        # 循环为每个分支构建网络
        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        # 进行多个分辨率层的融合
        if self.num_branches == 1:
            return None

        # 平行子网分支数
        num_branches = self.num_branches  # 3

        # 输入通道数
        num_inchannels = self.num_inchannels  # [32,64,128]
        fuse_layers = []

        # 为每个分支都创建对应的特征融合网络,如果multi_scale_output=1,则只需要一个特征融合网络
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                # 多种情况的讨论
                # 1.当前分支信息传递到上一分支(沿论文图示scale方向)的下一层(沿论文图示depth方向),
                # 进行上采样,分辨率加倍
                #         先使用1x1卷积将j分支的通道数变得和i分支一致,进而跟着BN,
                #    然后依据上采样因子将j分支分辨率上采样到和i分支分辨率相同,此处使用最近邻插值
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2**(j-i), mode='nearest')
                        )
                    )

                # 2.当前分支信息传递到当前分支(论文图示沿scale方向)的下一层(沿论文图示depth方向),
                # 不做任何操作,分辨率相同
                elif j == i:
                    fuse_layer.append(None)

                # 3.当前分支信息传递到下一分支(论文图示沿scale方向)的下一层(沿论文图示depth方向),
                # 进行下采样，分辨率减半
                # 里面包含了一个双层循环，要根据下采样的尺度决定循环的次数:
                #       当i-j > 1时，两个分支的分辨率差了不止二倍，此时还是两倍两倍往上采样，
                #       例如i-j = 2时，j分支的分辨率比i分支大4倍，就需要上采样两次，循环次数就是2；
                else:
                    conv3x3s = []
                    for k in range(i-j):  # 假设是 stage3 ---> stage4 ---> range(2 - 0) == range(2)
                        if k == i - j - 1:  # k == 1
                            num_outchannels_conv3x3 = num_inchannels[i]  # 128
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],  # 64
                                        num_outchannels_conv3x3,  # 128
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                    # 不使用 ReLU 应该是保持特征多样性
                                )
                            )
                        else:  # k == 0
                            num_outchannels_conv3x3 = num_inchannels[j]  # 32
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],  # 32
                                        num_outchannels_conv3x3,  # 32
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                        # 此处结构看论文是如何描述 fuselayer 的
                        # 每次多尺度之间的加法运算都是从最上面的尺度开始往下加，所以y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])；
                        # 加到他自己的时候，不需要经过融合函数的处理，直接加，所以if i == j: y = y + x[j]
                        # 遇到不是最上面的尺度那个特征图或者它本身相同分辨率的那个特征图时，需要经过融合函数处理再加，
                        # 所以y = y + self.fuse_layers[i][j](x[j])。最后将ReLU激活后的融合(加法)特征append到x_fuse，x_fuse的长度等于1（单尺度输出）或者num_branches（多尺度输出）。
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

#         最后返回的是一个存储了每个分支对应的融合模块的二维数组,比如说两个分支中，
#    对于分支１，fuse_layers[0][0]=None,fuse_layers[0][1]=上采样的操作

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        # ｘ表示每个分支输入的特征，如果有两个分支，则ｘ就是一个二维数组,
        # x[0]和x[1]就是两个输入分支的特征.

        # 如果只有一个分支就直接返回，不做任何融合
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        # 有多个分支的时候，对每个分支都先用_make_branch函数生成主特征网络，再将特定的网络特征进行融合
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        # 对每个分支交换信息，进行融合
        for i in range(len(self.fuse_layers)):
            # 循环融合多个分支的输出信息,当作输入,进行下一轮融合
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                # 进行特征融合，举个例子，运行到分支一的时候，
                # self.fuse_layers[i][0](x[0])先生成对于分支一的融合操作，
                # for循环得到每个分支对于分支一的采样结果【当i=j，就是分支一本身不进行任何操作直接cat，
                # i>j的时候就是分支２对于分支一来说要进行上采样，然后cat得到结果】，并cat得到最后的输出
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        # 输出的是最后融合的特征【例如两个分支的时候，输出分支一＋分支二的上采样和分支一的下采样＋分支二】
        return x_fuse


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class PoseHighResolutionNet(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        # 初始输入维度为64

        extra = cfg.MODEL.EXTRA
        # 存储 cfg 文件中的配置信息

        super(PoseHighResolutionNet, self).__init__()
        # 显示对PoseHighResolutionNet进行初始化（覆盖nn.Module的__init__()）
        # super() lets you avoid referring to the base class explicitly,
        # which can be nice. But the main advantage comes with multiple inheritance, where all sorts of fun stuff can happen.

        # stem net 获取原始特征图N11
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        # output-->256 channels


        # stage 2
        self.stage2_cfg = cfg['MODEL']['EXTRA']['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        # [32,64]
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        # BASIC
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)


        # stage3
        self.stage3_cfg = cfg['MODEL']['EXTRA']['STAGE3']
        
        # NUM_MODULES: 4
        # NUM_BRANCHES: 3
        # BLOCK: BASIC
        # NUM_BLOCKS:
        # - 4
        # - 4
        # - 4
        # NUM_CHANNELS:
        # - 32
        # - 64
        # - 128
        # FUSE_METHOD: SUM
        
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        # [32,64,128]
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        # BASIC
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)


        # stage4
        self.stage4_cfg = cfg['MODEL']['EXTRA']['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        # [32,64,128,256]
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        # BASIC
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)

        # 对最终的特征图混合之后进行一次卷积, 预测人体关键点的heatmap
        self.final_layer = nn.Conv2d(
            in_channels=pre_stage_channels[0],  # 64
            out_channels=cfg.MODEL.NUM_JOINTS,  # 17
            kernel_size=extra.FINAL_CONV_KERNEL,  # 1
            stride=1,
            padding=1 if extra.FINAL_CONV_KERNEL == 3 else 0  # 0
        )

        # 预测人体关键点的heatmap
        self.pretrained_layers = cfg['MODEL']['EXTRA']['PRETRAINED_LAYERS']

    def _make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):
        """
           :param num_channels_pre_layer: 上一个stage平行网络的输出通道数目,为一个list,
               stage=2时, num_channels_pre_layer = [256]
               stage=3时, num_channels_pre_layer = [32,64] //以这个为例子阅读
               stage=4时, num_channels_pre_layer = [32,64,128]
           :param num_channels_cur_layer:
               stage=2时, num_channels_cur_layer = [32,64]
               stage=3时, num_channels_cur_layer = [32,64,128] //**
               stage=4时, num_channels_cur_layer = [32,64,128,256]
        """

        num_branches_cur = len(num_channels_cur_layer)  # 3
        num_branches_pre = len(num_channels_pre_layer)  # 2

        transition_layers = []  # a list，alterable

        # 对stage的每个分支进行处理
        for i in range(num_branches_cur):  # 0，1，2

            # 若不为最后一个分支
            if i < num_branches_pre:
                # 当前层的输入通道和输出通道数不相等
                # 如果branches_cur通道数！=branches_pre通道数，那么就要用一个cnn网络改变通道数
                # 注意这个cnn是不会改变特征图的shape
                # 在stage1中，pre通道数是256，cur通道数为32，所以要添加这一层cnn改变通道数
                # 所以transition_layers第一层为
                # conv2d(256,32,3,1,1)
                # batchnorm2d(32)
                # relu
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            # 通过卷积对通道数进行变换
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            # https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
                            # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
                            # dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)

                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            # https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
                            # torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True,
                            # track_running_stats=True, device=None, dtype=None)

                            nn.ReLU(inplace=True)
                            # https://pytorch.org/docs/stable/generated/torch.nn.ReLU.html
                            # inplace – can optionally do the operation in-place. Default: False
                        )
                        # https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html
                        # *args ---> 表示任意多个无名参数
                        # 简单来讲就是将前一个操作的output作为下一个操作的input
                    )
                # 通道数相等，不做处理
                else:
                    transition_layers.append(None)
            # 为最后一个分支,则再新建一个分支且该分支分辨率会减少一半
            else:
                conv3x3s = []
                for j in range(i+1-num_branches_pre):  # 0
                    inchannels = num_channels_pre_layer[-1]  # 最后一个channel数 64
                    outchannels = num_channels_cur_layer[i] \
                        if j == i-num_branches_pre else inchannels
                    # i-num_branches_pre == 0 == j ---> 128
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)
        # https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html
        # ModuleList can act as an iterable, or be indexed using ints，如字面意思就是 a list of modules

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        """
            当stage=2时: num_inchannels=[32,64]           multi_scale_output=Ture
            当stage=3时: num_inchannels=[32,64,128]       multi_scale_output=Ture
            当stage=4时: num_inchannels=[32,64,128,256]   multi_scale_output=False
        """

        # 当stage=2,3,4时,num_modules分别为：1,4,3
        # 表示HighResolutionModule（平行之网络交换信息模块）模块的数目
        num_modules = layer_config['NUM_MODULES']

        # 当stage=2,3,4时,num_branches分别为：2,3,4,表示每个stage平行网络的数目
        num_branches = layer_config['NUM_BRANCHES']

        # 当stage=2,3,4时,num_blocks分别为：[4,4], [4,4,4], [4,4,4,4],
        # 表示每个stage blocks(BasicBlock或者BasicBlock)的数目
        num_blocks = layer_config['NUM_BLOCKS']

        # 当stage=2,3,4时,num_channels分别为：[32,64],[32,64,128],[32,64,128,256]
        # 在对应stage, 对应每个平行子网络的输出通道数
        num_channels = layer_config['NUM_CHANNELS']

        # HRNet-W32的所有stage的block均为BasicBlock
        block = blocks_dict[layer_config['BLOCK']]

        # HRNet-W32的所有stage的FUSE_METHOD(融合方式)均为SUM
        fuse_method = layer_config['FUSE_METHOD']

        modules = []

        # 根据num_modules的数目创建HighResolutionModule
        for i in range(num_modules):
            # multi_scale_output is only used last module
            # 为什只对最后的 module 使用 multi_scale_output
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            # 根据参数,添加HighResolutionModule
            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )

            # 获得最后一个HighResolutionModule的输出通道数
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        # 最初的数据特征提取基础部分
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        # 开始stage2的时候，第一次特征融合，先使用transition层得到特征融合部分的输入特征，就是将原有的一个分支，变为两个分支
        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)

        # 输入transition层得到的特征，进行特征融合得到特征融合后的输出，输入给第二个transition层得到stage3的输入特征，后面就类推
        y_list = self.stage2(x_list)

        # stage3
        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        # stage4
        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)

        # 这里只要stage４部分输出特征的第一个，视情况对 head 有不同的应用和处理方式，具体看论文描述
        x = self.final_layer(y_list[0])

        return x

    # 初始化权重 W
    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Return whether an object is an instance of a class or of a subclass thereof.
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                # Fills the input Tensor with values drawn from the normal distribution
                # tensor – an n-dimensional torch.Tensor
                # mean – the mean of the normal distribution
                # std – the standard deviation of the normal distribution
                # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_

                for name, _ in m.named_parameters():
                    # named_parameters()
                    # Returns an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
                        # https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.constant_
                        #

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                   or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED)

    return model
