import os
import sys
import math
from time import time
from functools import partial

import paddle
import paddle.nn as nn

from models.aspp import ASPP
from models.lib.modules import SynchronizedBatchNorm2d
from models.index import HolisticIndexBlock, DepthwiseO2OIndexBlock, DepthwiseM2OIndexBlock
from models.decoder import *
from models.conv import *

try:
    from urllib import urlretrieve
except ImportError:
    from urllib.request import urlretrieve

MODEL_URLS = {
    'mobilenetv2': 'https://paddle-imagenet-models-name.bj.bcebos.com/dygraph/MobileNetV2_ssld_pretrained.pdparams',
}

CORRESP_NAME = {
    # layer0
    "features.0.0.weight": "layer0.0.weight",
    "features.0.1.weight": "layer0.1.weight",
    "features.0.1.bias": "layer0.1.bias",
    "features.0.1.running_mean": "layer0.1.running_mean",
    "features.0.1.running_var": "layer0.1.running_var",
    # layer1
    "features.1.conv.0.weight": "layer1.0.conv.0.weight",
    "features.1.conv.1.weight": "layer1.0.conv.1.weight",
    "features.1.conv.1.bias": "layer1.0.conv.1.bias",
    "features.1.conv.1.running_mean": "layer1.0.conv.1.running_mean",
    "features.1.conv.1.running_var": "layer1.0.conv.1.running_var",
    "features.1.conv.3.weight": "layer1.0.conv.3.weight",
    "features.1.conv.4.weight": "layer1.0.conv.4.weight",
    "features.1.conv.4.bias": "layer1.0.conv.4.bias",
    "features.1.conv.4.running_mean": "layer1.0.conv.4.running_mean",
    "features.1.conv.4.running_var": "layer1.0.conv.4.running_var",
    # layer2
    "features.2.conv.0.weight": "layer2.0.conv.0.weight",
    "features.2.conv.1.weight": "layer2.0.conv.1.weight",
    "features.2.conv.1.bias": "layer2.0.conv.1.bias",
    "features.2.conv.1.running_mean": "layer2.0.conv.1.running_mean",
    "features.2.conv.1.running_var": "layer2.0.conv.1.running_var",
    "features.2.conv.3.weight": "layer2.0.conv.3.weight",
    "features.2.conv.4.weight": "layer2.0.conv.4.weight",
    "features.2.conv.4.bias": "layer2.0.conv.4.bias",
    "features.2.conv.4.running_mean": "layer2.0.conv.4.running_mean",
    "features.2.conv.4.running_var": "layer2.0.conv.4.running_var",
    "features.2.conv.6.weight": "layer2.0.conv.6.weight",
    "features.2.conv.7.weight": "layer2.0.conv.7.weight",
    "features.2.conv.7.bias": "layer2.0.conv.7.bias",
    "features.2.conv.7.running_mean": "layer2.0.conv.7.running_mean",
    "features.2.conv.7.running_var": "layer2.0.conv.7.running_var",

    "features.3.conv.0.weight": "layer2.1.conv.0.weight",
    "features.3.conv.1.weight": "layer2.1.conv.1.weight",
    "features.3.conv.1.bias": "layer2.1.conv.1.bias",
    "features.3.conv.1.running_mean": "layer2.1.conv.1.running_mean",
    "features.3.conv.1.running_var": "layer2.1.conv.1.running_var",
    "features.3.conv.3.weight": "layer2.1.conv.3.weight",
    "features.3.conv.4.weight": "layer2.1.conv.4.weight",
    "features.3.conv.4.bias": "layer2.1.conv.4.bias",
    "features.3.conv.4.running_mean": "layer2.1.conv.4.running_mean",
    "features.3.conv.4.running_var": "layer2.1.conv.4.running_var",
    "features.3.conv.6.weight": "layer2.1.conv.6.weight",
    "features.3.conv.7.weight": "layer2.1.conv.7.weight",
    "features.3.conv.7.bias": "layer2.1.conv.7.bias",
    "features.3.conv.7.running_mean": "layer2.1.conv.7.running_mean",
    "features.3.conv.7.running_var": "layer2.1.conv.7.running_var",
    # layer3
    "features.4.conv.0.weight": "layer3.0.conv.0.weight",
    "features.4.conv.1.weight": "layer3.0.conv.1.weight",
    "features.4.conv.1.bias": "layer3.0.conv.1.bias",
    "features.4.conv.1.running_mean": "layer3.0.conv.1.running_mean",
    "features.4.conv.1.running_var": "layer3.0.conv.1.running_var",
    "features.4.conv.3.weight": "layer3.0.conv.3.weight",
    "features.4.conv.4.weight": "layer3.0.conv.4.weight",
    "features.4.conv.4.bias": "layer3.0.conv.4.bias",
    "features.4.conv.4.running_mean": "layer3.0.conv.4.running_mean",
    "features.4.conv.4.running_var": "layer3.0.conv.4.running_var",
    "features.4.conv.6.weight": "layer3.0.conv.6.weight",
    "features.4.conv.7.weight": "layer3.0.conv.7.weight",
    "features.4.conv.7.bias": "layer3.0.conv.7.bias",
    "features.4.conv.7.running_mean": "layer3.0.conv.7.running_mean",
    "features.4.conv.7.running_var": "layer3.0.conv.7.running_var",

    "features.5.conv.0.weight": "layer3.1.conv.0.weight",
    "features.5.conv.1.weight": "layer3.1.conv.1.weight",
    "features.5.conv.1.bias": "layer3.1.conv.1.bias",
    "features.5.conv.1.running_mean": "layer3.1.conv.1.running_mean",
    "features.5.conv.1.running_var": "layer3.1.conv.1.running_var",
    "features.5.conv.3.weight": "layer3.1.conv.3.weight",
    "features.5.conv.4.weight": "layer3.1.conv.4.weight",
    "features.5.conv.4.bias": "layer3.1.conv.4.bias",
    "features.5.conv.4.running_mean": "layer3.1.conv.4.running_mean",
    "features.5.conv.4.running_var": "layer3.1.conv.4.running_var",
    "features.5.conv.6.weight": "layer3.1.conv.6.weight",
    "features.5.conv.7.weight": "layer3.1.conv.7.weight",
    "features.5.conv.7.bias": "layer3.1.conv.7.bias",
    "features.5.conv.7.running_mean": "layer3.1.conv.7.running_mean",
    "features.5.conv.7.running_var": "layer3.1.conv.7.running_var",

    "features.6.conv.0.weight": "layer3.2.conv.0.weight",
    "features.6.conv.1.weight": "layer3.2.conv.1.weight",
    "features.6.conv.1.bias": "layer3.2.conv.1.bias",
    "features.6.conv.1.running_mean": "layer3.2.conv.1.running_mean",
    "features.6.conv.1.running_var": "layer3.2.conv.1.running_var",
    "features.6.conv.3.weight": "layer3.2.conv.3.weight",
    "features.6.conv.4.weight": "layer3.2.conv.4.weight",
    "features.6.conv.4.bias": "layer3.2.conv.4.bias",
    "features.6.conv.4.running_mean": "layer3.2.conv.4.running_mean",
    "features.6.conv.4.running_var": "layer3.2.conv.4.running_var",
    "features.6.conv.6.weight": "layer3.2.conv.6.weight",
    "features.6.conv.7.weight": "layer3.2.conv.7.weight",
    "features.6.conv.7.bias": "layer3.2.conv.7.bias",
    "features.6.conv.7.running_mean": "layer3.2.conv.7.running_mean",
    "features.6.conv.7.running_var": "layer3.2.conv.7.running_var",
    # layer4
    "features.7.conv.0.weight": "layer4.0.conv.0.weight",
    "features.7.conv.1.weight": "layer4.0.conv.1.weight",
    "features.7.conv.1.bias": "layer4.0.conv.1.bias",
    "features.7.conv.1.running_mean": "layer4.0.conv.1.running_mean",
    "features.7.conv.1.running_var": "layer4.0.conv.1.running_var",
    "features.7.conv.3.weight": "layer4.0.conv.3.weight",
    "features.7.conv.4.weight": "layer4.0.conv.4.weight",
    "features.7.conv.4.bias": "layer4.0.conv.4.bias",
    "features.7.conv.4.running_mean": "layer4.0.conv.4.running_mean",
    "features.7.conv.4.running_var": "layer4.0.conv.4.running_var",
    "features.7.conv.6.weight": "layer4.0.conv.6.weight",
    "features.7.conv.7.weight": "layer4.0.conv.7.weight",
    "features.7.conv.7.bias": "layer4.0.conv.7.bias",
    "features.7.conv.7.running_mean": "layer4.0.conv.7.running_mean",
    "features.7.conv.7.running_var": "layer4.0.conv.7.running_var",

    "features.8.conv.0.weight": "layer4.1.conv.0.weight",
    "features.8.conv.1.weight": "layer4.1.conv.1.weight",
    "features.8.conv.1.bias": "layer4.1.conv.1.bias",
    "features.8.conv.1.running_mean": "layer4.1.conv.1.running_mean",
    "features.8.conv.1.running_var": "layer4.1.conv.1.running_var",
    "features.8.conv.3.weight": "layer4.1.conv.3.weight",
    "features.8.conv.4.weight": "layer4.1.conv.4.weight",
    "features.8.conv.4.bias": "layer4.1.conv.4.bias",
    "features.8.conv.4.running_mean": "layer4.1.conv.4.running_mean",
    "features.8.conv.4.running_var": "layer4.1.conv.4.running_var",
    "features.8.conv.6.weight": "layer4.1.conv.6.weight",
    "features.8.conv.7.weight": "layer4.1.conv.7.weight",
    "features.8.conv.7.bias": "layer4.1.conv.7.bias",
    "features.8.conv.7.running_mean": "layer4.1.conv.7.running_mean",
    "features.8.conv.7.running_var": "layer4.1.conv.7.running_var",

    "features.9.conv.0.weight": "layer4.2.conv.0.weight",
    "features.9.conv.1.weight": "layer4.2.conv.1.weight",
    "features.9.conv.1.bias": "layer4.2.conv.1.bias",
    "features.9.conv.1.running_mean": "layer4.2.conv.1.running_mean",
    "features.9.conv.1.running_var": "layer4.2.conv.1.running_var",
    "features.9.conv.3.weight": "layer4.2.conv.3.weight",
    "features.9.conv.4.weight": "layer4.2.conv.4.weight",
    "features.9.conv.4.bias": "layer4.2.conv.4.bias",
    "features.9.conv.4.running_mean": "layer4.2.conv.4.running_mean",
    "features.9.conv.4.running_var": "layer4.2.conv.4.running_var",
    "features.9.conv.6.weight": "layer4.2.conv.6.weight",
    "features.9.conv.7.weight": "layer4.2.conv.7.weight",
    "features.9.conv.7.bias": "layer4.2.conv.7.bias",
    "features.9.conv.7.running_mean": "layer4.2.conv.7.running_mean",
    "features.9.conv.7.running_var": "layer4.2.conv.7.running_var",

    "features.10.conv.0.weight": "layer4.3.conv.0.weight",
    "features.10.conv.1.weight": "layer4.3.conv.1.weight",
    "features.10.conv.1.bias": "layer4.3.conv.1.bias",
    "features.10.conv.1.running_mean": "layer4.3.conv.1.running_mean",
    "features.10.conv.1.running_var": "layer4.3.conv.1.running_var",
    "features.10.conv.3.weight": "layer4.3.conv.3.weight",
    "features.10.conv.4.weight": "layer4.3.conv.4.weight",
    "features.10.conv.4.bias": "layer4.3.conv.4.bias",
    "features.10.conv.4.running_mean": "layer4.3.conv.4.running_mean",
    "features.10.conv.4.running_var": "layer4.3.conv.4.running_var",
    "features.10.conv.6.weight": "layer4.3.conv.6.weight",
    "features.10.conv.7.weight": "layer4.3.conv.7.weight",
    "features.10.conv.7.bias": "layer4.3.conv.7.bias",
    "features.10.conv.7.running_mean": "layer4.3.conv.7.running_mean",
    "features.10.conv.7.running_var": "layer4.3.conv.7.running_var",
    # layer5
    "features.11.conv.0.weight": "layer5.0.conv.0.weight",
    "features.11.conv.1.weight": "layer5.0.conv.1.weight",
    "features.11.conv.1.bias": "layer5.0.conv.1.bias",
    "features.11.conv.1.running_mean": "layer5.0.conv.1.running_mean",
    "features.11.conv.1.running_var": "layer5.0.conv.1.running_var",
    "features.11.conv.3.weight": "layer5.0.conv.3.weight",
    "features.11.conv.4.weight": "layer5.0.conv.4.weight",
    "features.11.conv.4.bias": "layer5.0.conv.4.bias",
    "features.11.conv.4.running_mean": "layer5.0.conv.4.running_mean",
    "features.11.conv.4.running_var": "layer5.0.conv.4.running_var",
    "features.11.conv.6.weight": "layer5.0.conv.6.weight",
    "features.11.conv.7.weight": "layer5.0.conv.7.weight",
    "features.11.conv.7.bias": "layer5.0.conv.7.bias",
    "features.11.conv.7.running_mean": "layer5.0.conv.7.running_mean",
    "features.11.conv.7.running_var": "layer5.0.conv.7.running_var",

    "features.12.conv.0.weight": "layer5.1.conv.0.weight",
    "features.12.conv.1.weight": "layer5.1.conv.1.weight",
    "features.12.conv.1.bias": "layer5.1.conv.1.bias",
    "features.12.conv.1.running_mean": "layer5.1.conv.1.running_mean",
    "features.12.conv.1.running_var": "layer5.1.conv.1.running_var",
    "features.12.conv.3.weight": "layer5.1.conv.3.weight",
    "features.12.conv.4.weight": "layer5.1.conv.4.weight",
    "features.12.conv.4.bias": "layer5.1.conv.4.bias",
    "features.12.conv.4.running_mean": "layer5.1.conv.4.running_mean",
    "features.12.conv.4.running_var": "layer5.1.conv.4.running_var",
    "features.12.conv.6.weight": "layer5.1.conv.6.weight",
    "features.12.conv.7.weight": "layer5.1.conv.7.weight",
    "features.12.conv.7.bias": "layer5.1.conv.7.bias",
    "features.12.conv.7.running_mean": "layer5.1.conv.7.running_mean",
    "features.12.conv.7.running_var": "layer5.1.conv.7.running_var",

    "features.13.conv.0.weight": "layer5.2.conv.0.weight",
    "features.13.conv.1.weight": "layer5.2.conv.1.weight",
    "features.13.conv.1.bias": "layer5.2.conv.1.bias",
    "features.13.conv.1.running_mean": "layer5.2.conv.1.running_mean",
    "features.13.conv.1.running_var": "layer5.2.conv.1.running_var",
    "features.13.conv.3.weight": "layer5.2.conv.3.weight",
    "features.13.conv.4.weight": "layer5.2.conv.4.weight",
    "features.13.conv.4.bias": "layer5.2.conv.4.bias",
    "features.13.conv.4.running_mean": "layer5.2.conv.4.running_mean",
    "features.13.conv.4.running_var": "layer5.2.conv.4.running_var",
    "features.13.conv.6.weight": "layer5.2.conv.6.weight",
    "features.13.conv.7.weight": "layer5.2.conv.7.weight",
    "features.13.conv.7.bias": "layer5.2.conv.7.bias",
    "features.13.conv.7.running_mean": "layer5.2.conv.7.running_mean",
    "features.13.conv.7.running_var": "layer5.2.conv.7.running_var",
    # layer6
    "features.14.conv.0.weight": "layer6.0.conv.0.weight",
    "features.14.conv.1.weight": "layer6.0.conv.1.weight",
    "features.14.conv.1.bias": "layer6.0.conv.1.bias",
    "features.14.conv.1.running_mean": "layer6.0.conv.1.running_mean",
    "features.14.conv.1.running_var": "layer6.0.conv.1.running_var",
    "features.14.conv.3.weight": "layer6.0.conv.3.weight",
    "features.14.conv.4.weight": "layer6.0.conv.4.weight",
    "features.14.conv.4.bias": "layer6.0.conv.4.bias",
    "features.14.conv.4.running_mean": "layer6.0.conv.4.running_mean",
    "features.14.conv.4.running_var": "layer6.0.conv.4.running_var",
    "features.14.conv.6.weight": "layer6.0.conv.6.weight",
    "features.14.conv.7.weight": "layer6.0.conv.7.weight",
    "features.14.conv.7.bias": "layer6.0.conv.7.bias",
    "features.14.conv.7.running_mean": "layer6.0.conv.7.running_mean",
    "features.14.conv.7.running_var": "layer6.0.conv.7.running_var",

    "features.15.conv.0.weight": "layer6.1.conv.0.weight",
    "features.15.conv.1.weight": "layer6.1.conv.1.weight",
    "features.15.conv.1.bias": "layer6.1.conv.1.bias",
    "features.15.conv.1.running_mean": "layer6.1.conv.1.running_mean",
    "features.15.conv.1.running_var": "layer6.1.conv.1.running_var",
    "features.15.conv.3.weight": "layer6.1.conv.3.weight",
    "features.15.conv.4.weight": "layer6.1.conv.4.weight",
    "features.15.conv.4.bias": "layer6.1.conv.4.bias",
    "features.15.conv.4.running_mean": "layer6.1.conv.4.running_mean",
    "features.15.conv.4.running_var": "layer6.1.conv.4.running_var",
    "features.15.conv.6.weight": "layer6.1.conv.6.weight",
    "features.15.conv.7.weight": "layer6.1.conv.7.weight",
    "features.15.conv.7.bias": "layer6.1.conv.7.bias",
    "features.15.conv.7.running_mean": "layer6.1.conv.7.running_mean",
    "features.15.conv.7.running_var": "layer6.1.conv.7.running_var",

    "features.16.conv.0.weight": "layer6.2.conv.0.weight",
    "features.16.conv.1.weight": "layer6.2.conv.1.weight",
    "features.16.conv.1.bias": "layer6.2.conv.1.bias",
    "features.16.conv.1.running_mean": "layer6.2.conv.1.running_mean",
    "features.16.conv.1.running_var": "layer6.2.conv.1.running_var",
    "features.16.conv.3.weight": "layer6.2.conv.3.weight",
    "features.16.conv.4.weight": "layer6.2.conv.4.weight",
    "features.16.conv.4.bias": "layer6.2.conv.4.bias",
    "features.16.conv.4.running_mean": "layer6.2.conv.4.running_mean",
    "features.16.conv.4.running_var": "layer6.2.conv.4.running_var",
    "features.16.conv.6.weight": "layer6.2.conv.6.weight",
    "features.16.conv.7.weight": "layer6.2.conv.7.weight",
    "features.16.conv.7.bias": "layer6.2.conv.7.bias",
    "features.16.conv.7.running_mean": "layer6.2.conv.7.running_mean",
    "features.16.conv.7.running_var": "layer6.2.conv.7.running_var",
    # layer7
    "features.17.conv.0.weight": "layer7.0.conv.0.weight",
    "features.17.conv.1.weight": "layer7.0.conv.1.weight",
    "features.17.conv.1.bias": "layer7.0.conv.1.bias",
    "features.17.conv.1.running_mean": "layer7.0.conv.1.running_mean",
    "features.17.conv.1.running_var": "layer7.0.conv.1.running_var",
    "features.17.conv.3.weight": "layer7.0.conv.3.weight",
    "features.17.conv.4.weight": "layer7.0.conv.4.weight",
    "features.17.conv.4.bias": "layer7.0.conv.4.bias",
    "features.17.conv.4.running_mean": "layer7.0.conv.4.running_mean",
    "features.17.conv.4.running_var": "layer7.0.conv.4.running_var",
    "features.17.conv.6.weight": "layer7.0.conv.6.weight",
    "features.17.conv.7.weight": "layer7.0.conv.7.weight",
    "features.17.conv.7.bias": "layer7.0.conv.7.bias",
    "features.17.conv.7.running_mean": "layer7.0.conv.7.running_mean",
    "features.17.conv.7.running_var": "layer7.0.conv.7.running_var",
}


def pred(inp, oup, conv_operator, k, batch_norm):
    # the last 1x1 convolutional layer is very important
    Conv2d = conv[conv_operator]
    return nn.Sequential(
        Conv2d(inp, oup, k, 1, batch_norm),
        nn.Conv2D(oup, oup, k, 1, padding=k // 2, bias_attr=False)
    )


class InvertedResidual(nn.Layer):  # 倒残差网络
    def __init__(self, inp, oup, stride, dilation, expand_ratio, batch_norm):  # expand_radio为扩展因子
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        BatchNorm2d = batch_norm

        hidden_dim = round(inp * expand_ratio)  # 卷积核个数
        self.use_res_connect = self.stride == 1 and inp == oup  # 判断是否使用捷径分支
        self.kernel_size = 3
        self.dilation = dilation

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2D(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias_attr=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw 1*1
                nn.Conv2D(inp, hidden_dim, 1, 1, 0, bias_attr=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # dw 要求groups=输入个数
                nn.Conv2D(hidden_dim, hidden_dim, 3, stride, 0, dilation, groups=hidden_dim, bias_attr=False),
                BatchNorm2d(hidden_dim),
                nn.ReLU6(),
                # pw-linear
                nn.Conv2D(hidden_dim, oup, 1, 1, 0, bias_attr=False),
                BatchNorm2d(oup),
            )

    def fixed_padding(self, inputs, kernel_size, dilation):
        kernel_size_effective = kernel_size + (kernel_size - 1) * (dilation - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        padded_inputs = F.pad(inputs, (pad_beg, pad_end, pad_beg, pad_end))
        return padded_inputs

    def _nostride_dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m._kernel_size == [3, 3]:
                m._dilation = [dilate, dilate]
                m._padding = dilate

    def forward(self, x):  # 正向传播过程，x为输入特征矩阵
        x_pad = self.fixed_padding(x, self.kernel_size, dilation=self.dilation)
        if self.use_res_connect:
            return x + self.conv(x_pad)
        else:
            return self.conv(x_pad)


#######################################################################################
# DeepLabv3+ B1
#######################################################################################
class MobileNetV2DeepLabv3Plus(nn.Layer):
    def __init__(
            self,
            output_stride=16,
            input_size=321,
            width_mult=1.,
            conv_operator='std_conv',
            decoder_kernel_size=5,
            apply_aspp=False,
            freeze_bn=False,
            sync_bn=False,
            **kwargs
    ):
        super(MobileNetV2DeepLabv3Plus, self).__init__()
        self.width_mult = width_mult

        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2D

        block = InvertedResidual
        decoder = DeepLabDecoder
        aspp = ASPP
        initial_channel = 32
        current_stride = 1
        rate = 1
        inverted_residual_setting = [
            # t, p, c, n, s, d
            [1, initial_channel, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        ### encoder ###
        # building the first layer
        assert input_size % output_stride == 1
        initial_channel = int(initial_channel * width_mult)
        self.layer0 = conv_bn(4, initial_channel, 3, 2, BatchNorm2d)
        current_stride *= 2
        # building bottleneck layers
        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            if current_stride == output_stride:
                inverted_residual_setting[i][4] = 1  # change stride
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s
        self.layer1 = self._build_layer(block, inverted_residual_setting[0], BatchNorm2d)
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], BatchNorm2d, downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], BatchNorm2d, downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], BatchNorm2d, downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4], BatchNorm2d)
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], BatchNorm2d, downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6], BatchNorm2d)

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()

        ### context aggregation ###
        self.dconv_pp = aspp(320, 256, output_stride=output_stride, batch_norm=BatchNorm2d)

        ### decoder ###
        self.decoder = decoder(conv_operator, decoder_kernel_size, batch_norm=BatchNorm2d)

        self.pred = nn.Sequential(
            nn.Conv2D(256, 1, 1, 1, padding=0, bias_attr=False),
            BatchNorm2d(1),
            nn.ReLU6(),
            nn.Conv2D(1, 1, 1, 1, padding=0, bias_attr=False)
        )

        self._init_weight()

    def _build_layer(self, block, layer_setting, batch_norm, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, batch_norm=batch_norm))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, batch_norm=batch_norm))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def forward(self, x):
        # encode
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)

        # pyramid pooling
        l = self.dconv_pp(l7)

        # decode
        l = self.decoder(l, l2)

        # prediction
        l = self.pred(l)
        l = F.interpolate(l, size=x.shape[2:], mode='bilinear', align_corners=True)

        return l

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2D):
                m.eval()

    def _init_weight(self):
        initlizer_1 = nn.initializer.Constant(1)
        initlizer_0 = nn.initializer.Constant(0)
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                initlizer = nn.initializer.KaimingUniform(None, 0, 'relu')
                initlizer(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                initlizer_1(m.weight)
                initlizer_0(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                initlizer_1(m.weight)
                initlizer_0(m.bias)


#######################################################################################
# RefineNet B2
#######################################################################################
class CRPBlock(nn.Layer):
    def __init__(self, inp, oup, n_stages, batch_norm):
        super(CRPBlock, self).__init__()
        BatchNorm2d = batch_norm
        for i in range(n_stages):
            setattr(
                self, '{}_{}'.format(i + 1, 'outvar_dimred'),
                conv_bn(inp if (i == 0) else oup, oup, 1, 1, BatchNorm2d)
            )
        self.stride = 1
        self.n_stages = n_stages
        self.maxpool = nn.MaxPool2D(kernel_size=5, stride=1, padding=2)

    def forward(self, x):
        top = x
        for i in range(self.n_stages):
            top = self.maxpool(top)
            top = getattr(self, '{}_{}'.format(i + 1, 'outvar_dimred'))(top)
            x = top + x
        return x


class MobileNetV2RefineNet(nn.Layer):
    def __init__(
            self,
            output_stride=16,
            input_size=321,
            width_mult=1.,
            conv_operator='std_conv',
            decoder_kernel_size=5,
            apply_aspp=False,
            freeze_bn=False,
            sync_bn=False,
            **kwargs
    ):
        super(MobileNetV2RefineNet, self).__init__()
        self.width_mult = width_mult

        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2D

        block = InvertedResidual
        initial_channel = 32
        current_stride = 1
        rate = 1
        inverted_residual_setting = [
            # t, p, c, n, s, d
            [1, initial_channel, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        ### encoder ###
        # building the first layer
        assert input_size % output_stride == 1
        initial_channel = int(initial_channel * width_mult)
        self.layer0 = conv_bn(4, initial_channel, 3, 2, BatchNorm2d)
        current_stride *= 2
        # building bottleneck layers
        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            if current_stride == output_stride:
                inverted_residual_setting[i][4] = 1  # change stride
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s
        self.layer1 = self._build_layer(block, inverted_residual_setting[0], BatchNorm2d)
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], BatchNorm2d, downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], BatchNorm2d, downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], BatchNorm2d, downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4], BatchNorm2d)
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], BatchNorm2d, downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6], BatchNorm2d)

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()

        # light-weight RefineNet
        self.dconv7 = conv_bn(320, 256, 1, 1, BatchNorm2d)
        self.dconv6 = conv_bn(160, 256, 1, 1, BatchNorm2d)
        self.dconv5 = conv_bn(96, 256, 1, 1, BatchNorm2d)
        self.dconv4 = conv_bn(64, 256, 1, 1, BatchNorm2d)
        self.dconv3 = conv_bn(32, 256, 1, 1, BatchNorm2d)
        self.dconv2 = conv_bn(24, 256, 1, 1, BatchNorm2d)
        self.dconv_crp4 = self._make_crp(256, 256, 4, BatchNorm2d)
        self.dconv_crp3 = self._make_crp(256, 256, 4, BatchNorm2d)
        self.dconv_crp2 = self._make_crp(256, 256, 4, BatchNorm2d)
        self.dconv_crp1 = self._make_crp(256, 256, 4, BatchNorm2d)
        self.dconv_adapt4 = conv_bn(256, 256, 1, 1, BatchNorm2d)
        self.dconv_adapt3 = conv_bn(256, 256, 1, 1, BatchNorm2d)
        self.dconv_adapt2 = conv_bn(256, 256, 1, 1, BatchNorm2d)
        self.relu = nn.ReLU6()

        self.pred = pred(256, 1, conv_operator, k=decoder_kernel_size, batch_norm=BatchNorm2d)

        self._init_weight()

    def _build_layer(self, block, layer_setting, batch_norm, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, batch_norm=batch_norm))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, batch_norm=batch_norm))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def _make_crp(self, in_planes, out_planes, stages, batch_norm):
        layers = [CRPBlock(in_planes, out_planes, stages, batch_norm)]
        return nn.Sequential(*layers)

    def forward(self, x):
        # encode
        l0 = self.layer0(x)
        l1 = self.layer1(l0)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)
        l5 = self.layer5(l4)
        l6 = self.layer6(l5)
        l7 = self.layer7(l6)

        # decode
        l7 = self.dconv7(l7)
        l6 = self.dconv6(l6)
        l6 = self.relu(l7 + l6)
        l6 = self.dconv_crp4(l6)
        l6 = self.dconv_adapt4(l6)
        l6 = F.interpolate(l6, size=l5.shape[2:], mode='bilinear', align_corners=True)

        l5 = self.dconv5(l5)
        l4 = self.dconv4(l4)
        l4 = self.relu(l4 + l5 + l6)
        l4 = self.dconv_crp3(l4)
        l4 = self.dconv_adapt3(l4)
        l4 = F.interpolate(l4, size=l3.shape[2:], mode='bilinear', align_corners=True)

        l3 = self.dconv3(l3)
        l3 = self.relu(l4 + l3)
        l3 = self.dconv_crp2(l3)
        l3 = self.dconv_adapt2(l3)
        l3 = F.interpolate(l3, size=l2.shape[2:], mode='bilinear', align_corners=True)

        l2 = self.dconv2(l2)
        l2 = self.relu(l2 + l3)
        l2 = self.dconv_crp1(l2)

        # prediction
        l = self.pred(l2)
        l = F.interpolate(l, size=x.shape[2:], mode='bilinear', align_corners=True)

        return l

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2D):
                m.eval()

    def _init_weight(self):
        initlizer_1 = nn.initializer.Constant(1)
        initlizer_0 = nn.initializer.Constant(0)
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                initlizer = nn.initializer.KaimingUniform(None, 0, 'relu')
                initlizer(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                initlizer_1(m.weight)
                initlizer_0(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                initlizer_1(m.weight)
                initlizer_0(m.bias)

#######################################################################################
# UNet B11
#######################################################################################
class MobileNetV2UNetDecoder(nn.Layer):
    def __init__(
            self,
            output_stride=32,
            input_size=320,
            width_mult=1.,
            conv_operator='std_conv',
            decoder_kernel_size=5,
            apply_aspp=False,
            freeze_bn=False,
            sync_bn=False,
            **kwargs
    ):
        super(MobileNetV2UNetDecoder, self).__init__()
        self.width_mult = width_mult
        self.output_stride = output_stride

        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2D

        block = InvertedResidual
        aspp = ASPP
        decoder_block = IndexedDecoder
        initial_channel = 32
        current_stride = 1
        rate = 1
        inverted_residual_setting = [
            # expand_ratio, input_chn, output_chn, num_blocks, stride, dilation
            [1, initial_channel, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        ### encoder ###
        # building the first layer
        assert input_size % output_stride == 0
        initial_channel = int(initial_channel * width_mult)
        self.layer0 = conv_bn(4, initial_channel, 3, 2, BatchNorm2d)
        self.layer0.apply(partial(self._stride, stride=1))  # set stride = 1
        current_stride *= 2
        # building bottleneck layers
        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            inverted_residual_setting[i][4] = 1  # change stride
            if current_stride == output_stride:
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s
        self.layer1 = self._build_layer(block, inverted_residual_setting[0], BatchNorm2d)
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], BatchNorm2d, downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], BatchNorm2d, downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], BatchNorm2d, downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4], BatchNorm2d)
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], BatchNorm2d, downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6], BatchNorm2d)

        if output_stride == 32:
            self.pool0 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool2 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool3 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool4 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool6 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
        elif output_stride == 16:
            self.pool0 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool2 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool3 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool4 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
        elif output_stride == 8:
            self.pool0 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool2 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)
            self.pool3 = nn.MaxPool2D((2, 2), stride=2, padding=0, return_mask=True)

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()

        ### context aggregation ###
        if apply_aspp:
            self.dconv_pp = aspp(int(320 * width_mult), int(160 * width_mult), output_stride=output_stride,
                                 batch_norm=BatchNorm2d, width_mult=width_mult)
        else:
            self.dconv_pp = conv_bn(int(320 * width_mult), int(160 * width_mult), 1, 1, BatchNorm2d)

        ### decoder ###
        self.decoder_layer6 = decoder_block(int(160 * width_mult) * 2, int(96 * width_mult),
                                            conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)
        self.decoder_layer5 = decoder_block(int(96 * width_mult) * 2, int(64 * width_mult), conv_operator=conv_operator,
                                            kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer4 = decoder_block(int(64 * width_mult) * 2, int(32 * width_mult), conv_operator=conv_operator,
                                            kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer3 = decoder_block(int(32 * width_mult) * 2, int(24 * width_mult), conv_operator=conv_operator,
                                            kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer2 = decoder_block(int(24 * width_mult) * 2, int(16 * width_mult), conv_operator=conv_operator,
                                            kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer1 = decoder_block(int(16 * width_mult) * 2, int(32 * width_mult), conv_operator=conv_operator,
                                            kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)
        self.decoder_layer0 = decoder_block(int(32 * width_mult) * 2, int(32 * width_mult), conv_operator=conv_operator,
                                            kernel_size=decoder_kernel_size, batch_norm=BatchNorm2d)

        self.pred = pred(int(32 * width_mult), 1, conv_operator, k=decoder_kernel_size, batch_norm=BatchNorm2d)

        self._init_weight()

    def _build_layer(self, block, layer_setting, batch_norm, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, batch_norm=batch_norm))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, batch_norm=batch_norm))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def _dilate(self, m, dilate):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m._kernel_size == [3, 3]:
                m._dilation = [dilate, dilate]
                m._padding = dilate

    def _stride(self, m, stride):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m._kernel_size == [3, 3]:
                m._stride = [stride, stride]
                return

    def forward(self, x):
        # encode
        l0 = self.layer0(x)  # 4x320x320
        l0p, idx0 = self.pool0(l0)  # 32x160x160

        l1 = self.layer1(l0p)  # 16x160x160
        l2 = self.layer2(l1)  # 24x160x160
        l2p, idx2 = self.pool2(l2)  # 24x80x80

        l3 = self.layer3(l2p)  # 32x80x80
        l3p, idx3 = self.pool3(l3)  # 32x40x40

        l4 = self.layer4(l3p)  # 64x40x40
        if self.output_stride == 8:
            l4p, idx4 = l4, None
        else:
            l4p, idx4 = self.pool4(l4)  # 64x20x20

        l5 = self.layer5(l4p)  # 96x20x20

        l6 = self.layer6(l5)  # 160x20x20
        if self.output_stride == 32:
            l6p, idx6 = self.pool6(l6)  # 160x10x10
        elif self.output_stride == 16 or self.output_stride == 8:
            l6p, idx6 = l6, None
        else:
            raise NotImplementedError

        l7 = self.layer7(l6p)  # 320x10x10

        # pyramid pooling
        l = self.dconv_pp(l7)  # 160x10x10

        # decode
        l = self.decoder_layer6(l, l6, idx6)
        l = self.decoder_layer5(l, l5)
        l = self.decoder_layer4(l, l4, idx4)
        l = self.decoder_layer3(l, l3, idx3)
        l = self.decoder_layer2(l, l2, idx2)
        l = self.decoder_layer1(l, l1)
        l = self.decoder_layer0(l, l0, idx0)

        l = self.pred(l)

        return l

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2D):
                m.eval()

    def _init_weight(self):
        initlizer_1 = nn.initializer.Constant(1)
        initlizer_0 = nn.initializer.Constant(0)
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                initlizer = nn.initializer.KaimingUniform(None, 0, 'relu')
                initlizer(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                initlizer_1(m.weight)
                initlizer_0(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                initlizer_1(m.weight)
                initlizer_0(m.bias)


#######################################################################################
# IndexNet
#######################################################################################
class MobileNetV2UNetDecoderIndexLearning(nn.Layer):
    def __init__(
            self,
            output_stride=32,
            input_size=320,
            width_mult=1.,
            conv_operator='std_conv',
            decoder_kernel_size=5,
            apply_aspp=False,
            freeze_bn=False,
            use_nonlinear=False,
            use_context=False,
            indexnet='holistic',
            index_mode='o2o',
            sync_bn=False
    ):
        super(MobileNetV2UNetDecoderIndexLearning, self).__init__()
        self.width_mult = width_mult
        self.output_stride = output_stride
        self.index_mode = index_mode

        BatchNorm2d = SynchronizedBatchNorm2d if sync_bn else nn.BatchNorm2D

        block = InvertedResidual
        aspp = ASPP
        decoder_block = IndexedUpsamlping

        if indexnet == 'holistic':
            index_block = HolisticIndexBlock
        elif indexnet == 'depthwise':
            if 'o2o' in index_mode:
                index_block = DepthwiseO2OIndexBlock
            elif 'm2o' in index_mode:
                index_block = DepthwiseM2OIndexBlock
            else:
                raise NameError
        else:
            raise NameError

        initial_channel = 32
        current_stride = 1
        rate = 1
        inverted_residual_setting = [
            # expand_ratio, input_chn, output_chn, num_blocks, stride, dilation
            [1, initial_channel, 16, 1, 1, 1],
            [6, 16, 24, 2, 2, 1],
            [6, 24, 32, 3, 2, 1],
            [6, 32, 64, 4, 2, 1],
            [6, 64, 96, 3, 1, 1],
            [6, 96, 160, 3, 2, 1],
            [6, 160, 320, 1, 1, 1],
        ]

        ### encoder ###
        # building the first layer
        # assert input_size % output_stride == 0
        initial_channel = int(initial_channel * width_mult)
        self.layer0 = conv_bn(4, initial_channel, 3, 2, BatchNorm2d)
        self.layer0.apply(partial(self._stride, stride=1))  # set stride = 1
        current_stride *= 2
        # building bottleneck layers
        for i, setting in enumerate(inverted_residual_setting):
            s = setting[4]
            inverted_residual_setting[i][4] = 1  # change stride
            if current_stride == output_stride:
                rate *= s
                inverted_residual_setting[i][5] = rate
            else:
                current_stride *= s
        self.layer1 = self._build_layer(block, inverted_residual_setting[0], BatchNorm2d)
        self.layer2 = self._build_layer(block, inverted_residual_setting[1], BatchNorm2d, downsample=True)
        self.layer3 = self._build_layer(block, inverted_residual_setting[2], BatchNorm2d, downsample=True)
        self.layer4 = self._build_layer(block, inverted_residual_setting[3], BatchNorm2d, downsample=True)
        self.layer5 = self._build_layer(block, inverted_residual_setting[4], BatchNorm2d)
        self.layer6 = self._build_layer(block, inverted_residual_setting[5], BatchNorm2d, downsample=True)
        self.layer7 = self._build_layer(block, inverted_residual_setting[6], BatchNorm2d)

        # freeze encoder batch norm layers
        if freeze_bn:
            self.freeze_bn()

        # define index blocks
        if output_stride == 32:
            self.index0 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index2 = index_block(24, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index3 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index4 = index_block(64, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index6 = index_block(160, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
        elif output_stride == 16:
            self.index0 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index2 = index_block(24, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index3 = index_block(32, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
            self.index4 = index_block(64, use_nonlinear=use_nonlinear, use_context=use_context, batch_norm=BatchNorm2d)
        else:
            raise NotImplementedError

        ### context aggregation ###
        if apply_aspp:
            self.dconv_pp = aspp(320, 160, output_stride=output_stride, batch_norm=BatchNorm2d)
        else:
            self.dconv_pp = conv_bn(320, 160, 1, 1, BatchNorm2d)

        ### decoder ###
        self.decoder_layer6 = decoder_block(160 * 2, 96, conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)
        self.decoder_layer5 = decoder_block(96 * 2, 64, conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)
        self.decoder_layer4 = decoder_block(64 * 2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)
        self.decoder_layer3 = decoder_block(32 * 2, 24, conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)
        self.decoder_layer2 = decoder_block(24 * 2, 16, conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)
        self.decoder_layer1 = decoder_block(16 * 2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)
        self.decoder_layer0 = decoder_block(32 * 2, 32, conv_operator=conv_operator, kernel_size=decoder_kernel_size,
                                            batch_norm=BatchNorm2d)

        self.pred = pred(32, 1, conv_operator, k=decoder_kernel_size, batch_norm=BatchNorm2d)

        self._init_weight()

    def _build_layer(self, block, layer_setting, batch_norm, downsample=False):
        t, p, c, n, s, d = layer_setting
        input_channel = int(p * self.width_mult)
        output_channel = int(c * self.width_mult)

        layers = []
        for i in range(n):
            if i == 0:
                d0 = d
                if downsample:
                    d0 = d // 2 if d > 1 else 1
                layers.append(block(input_channel, output_channel, s, d0, expand_ratio=t, batch_norm=batch_norm))
            else:
                layers.append(block(input_channel, output_channel, 1, d, expand_ratio=t, batch_norm=batch_norm))
            input_channel = output_channel

        return nn.Sequential(*layers)

    def _stride(self, m, stride):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            if m._kernel_size == [3, 3]:
                m._stride = [stride, stride]
                return

    def forward(self, x):
        # encode
        l0 = self.layer0(x)  # 4x320x320
        idx0_en, idx0_de = self.index0(l0)
        l0 = idx0_en * l0
        l0p = 4 * F.avg_pool2d(l0, (2, 2), stride=2)  # 32x160x160

        l1 = self.layer1(l0p)  # 16x160x160
        l2 = self.layer2(l1)  # 24x160x160
        idx2_en, idx2_de = self.index2(l2)
        l2 = idx2_en * l2
        l2p = 4 * F.avg_pool2d(l2, (2, 2), stride=2)  # 24x80x80

        l3 = self.layer3(l2p)  # 32x80x80
        idx3_en, idx3_de = self.index3(l3)
        l3 = idx3_en * l3
        l3p = 4 * F.avg_pool2d(l3, (2, 2), stride=2)  # 32x40x40

        l4 = self.layer4(l3p)  # 64x40x40
        idx4_en, idx4_de = self.index4(l4)
        l4 = idx4_en * l4
        l4p = 4 * F.avg_pool2d(l4, (2, 2), stride=2)  # 64x20x20

        l5 = self.layer5(l4p)  # 96x20x20
        l6 = self.layer6(l5)  # 160x20x20
        if self.output_stride == 32:
            idx6_en, idx6_de = self.index6(l6)
            l6 = idx6_en * l6
            l6p = 4 * F.avg_pool2d(l6, (2, 2), stride=2)  # 160x10x10
        elif self.output_stride == 16:
            l6p, idx6_de = l6, None

        l7 = self.layer7(l6p)  # 320x10x10

        # pyramid pooling
        l = self.dconv_pp(l7)  # 160x10x10

        # decode
        l = self.decoder_layer6(l, l6, idx6_de)
        l = self.decoder_layer5(l, l5)
        l = self.decoder_layer4(l, l4, idx4_de)
        l = self.decoder_layer3(l, l3, idx3_de)
        l = self.decoder_layer2(l, l2, idx2_de)
        l = self.decoder_layer1(l, l1)
        l = self.decoder_layer0(l, l0, idx0_de)

        l = self.pred(l)

        return l

    def freeze_bn(self):
        for m in self.sublayers():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2D):
                m.eval()

    def _init_weight(self):
        initlizer_1 = nn.initializer.Constant(1)
        initlizer_0 = nn.initializer.Constant(0)
        for m in self.sublayers():
            if isinstance(m, nn.Conv2D):
                initlizer = nn.initializer.KaimingUniform(None, 0, 'relu')
                initlizer(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                initlizer_1(m.weight)
                initlizer_0(m.bias)
            elif isinstance(m, nn.BatchNorm2D):
                initlizer_1(m.weight)
                initlizer_0(m.bias)


def mobilenetv2(pretrained=False, decoder='unet_style', **kwargs):
    """Constructs a MobileNet_V2 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    if decoder == 'unet_style':
        model = MobileNetV2UNetDecoder(**kwargs)
    elif decoder == 'indexnet':
        model = MobileNetV2UNetDecoderIndexLearning(**kwargs)
    elif decoder == 'deeplabv3+':
        model = MobileNetV2DeepLabv3Plus(**kwargs)
    elif decoder == 'refinenet':
        model = MobileNetV2RefineNet(**kwargs)
    else:
        raise NotImplementedError

    if pretrained:
        corresp_name = CORRESP_NAME
        model_dict = model.state_dict()
        pretrained_dict = load_url(MODEL_URLS['mobilenetv2'])
        for name in pretrained_dict:
            if name not in corresp_name:
                continue
            if corresp_name[name] not in model_dict.keys():
                continue
            if name == "features.0.0.weight":
                model_weight = model_dict[corresp_name[name]]
                assert model_weight.shape[1] == 4
                model_weight[:, 0:3, :, :] = pretrained_dict[name]
                model_weight[:, 3, :, :] = paddle.to_tensor(0)
                model_dict[corresp_name[name]] = model_weight
            else:
                model_dict[corresp_name[name]] = pretrained_dict[name]
        model.set_state_dict(model_dict)
    return model


def load_url(url, model_dir='./models/pretrained', map_location=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    filename = url.split('/')[-1]
    cached_file = os.path.join(model_dir, filename)
    if not os.path.exists(cached_file):
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        urlretrieve(url, cached_file)
    return paddle.load(cached_file)


if __name__ == "__main__":
    import numpy as np

    paddle.device.set_device('gpu:0')

    net = mobilenetv2(
        width_mult=1,
        pretrained=True,
        freeze_bn=True,
        sync_bn=False,
        apply_aspp=True,
        output_stride=32,
        conv_operator='std_conv',
        decoder_kernel_size=5,
        decoder='unet_style',
        indexnet='depthwise',
        index_mode='m2o',
        use_nonlinear=True,
        use_context=True,
    )
    net.eval()

    dump_x = paddle.randn((1, 4, 224, 224)).cuda()
    print(paddle.summary(net, input=dump_x))

    frame_rate = np.zeros((10, 1))
    for i in range(10):
        x = paddle.randn((1, 4, 320, 320)).cuda()
        paddle.device.cuda.synchronize()
        start = time()
        y = net(x)
        print(y)
        paddle.device.cuda.synchronize()
        end = time()
        running_frame_rate = 1 * float(1 / (end - start))
        frame_rate[i] = running_frame_rate
    print(np.mean(frame_rate))
    print(y.shape)
