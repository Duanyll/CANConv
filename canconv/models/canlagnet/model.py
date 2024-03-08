# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
# @Author  : Xiao Wu, LiangJian Deng
# @reference:
import torch
import torch.nn as nn
from torch.nn import functional as F
from canconv.layers.canconv import CANConv

from canconv.model.lagnet.model import LAGConv2D, LACRB

class CANCRB(nn.Module):
    def __init__(self, in_planes, cluster_num=32):
        super(CANCRB, self).__init__()
        self.conv1 = CANConv(in_planes, in_planes, cluster_num=cluster_num)
        self.relu1 = nn.ReLU()
        self.conv2 = CANConv(in_planes, in_planes, cluster_num=cluster_num)

    def forward(self, x, cache_indice=None):
        res, idx = self.conv1(x, cache_indice)
        res = self.relu1(res)
        res, _ = self.conv2(res, cache_indice, idx)
        x = x + res
        return x


class CANCRB_Down(nn.Module):
    def __init__(self, in_planes, cluster_num=32):
        super(CANCRB_Down, self).__init__()
        self.conv1 = LAGConv2D(in_planes, in_planes, 3, 1, 1, use_bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = CANConv(in_planes, in_planes, cluster_num=cluster_num)

    def forward(self, x, cache_indice=None, cluster_override=None):
        res = self.conv1(x)
        res = self.relu1(res)
        res, idx = self.conv2(x, cache_indice, cluster_override)
        x = x + res
        return x, idx


class CANCRB_Up(nn.Module):
    def __init__(self, in_planes, cluster_num=32):
        super(CANCRB_Up, self).__init__()
        self.conv1 = CANConv(in_planes, in_planes, cluster_num=cluster_num)
        self.relu1 = nn.ReLU()
        self.conv2 = LAGConv2D(in_planes, in_planes, 3, 1, 1, use_bias=True)

    def forward(self, x, cache_indice=None, cluster_override=None):
        res, idx = self.conv1(x, cache_indice, cluster_override)
        res = self.relu1(res)
        res = self.conv2(res)
        x = x + res
        return x, idx

# Proposed Network


class CANLAGNET(nn.Module):
    def __init__(self, spectral_num, cluster_num=32):
        super(CANLAGNET, self).__init__()

        self.head_conv = nn.Sequential(
            LAGConv2D(spectral_num+1, 32, 3, 1, 1, use_bias=True),
            nn.ReLU(inplace=True)
        )

        self.RB1 = LACRB(32)
        self.RB2 = CANCRB(32, cluster_num=cluster_num)
        self.RB3 = LACRB(32)
        self.RB4 = CANCRB(32, cluster_num=cluster_num)
        self.RB5 = LACRB(32)

        self.tail_conv = LAGConv2D(32, spectral_num, 3, 1, 1, use_bias=True)

    def forward(self, pan, lms, cache_indice=None):
        x = torch.cat([pan, lms], 1)
        x = self.head_conv(x)
        x = self.RB1(x)
        x = self.RB2(x, cache_indice)
        x = self.RB3(x)
        x = self.RB4(x, cache_indice)
        x = self.RB5(x)
        x = self.tail_conv(x)
        sr = lms + x
        return sr
