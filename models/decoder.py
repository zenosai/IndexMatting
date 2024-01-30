from matplotlib import pyplot as plt

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from models.lib.modules import SynchronizedBatchNorm2d

from models.conv import conv

class DeepLabDecoder(nn.Layer):
    def __init__(self, conv_operator='std_conv', kernel_size=5, batch_norm=SynchronizedBatchNorm2d):
        super(DeepLabDecoder, self).__init__()
        Conv2d = conv[conv_operator]
        BatchNorm2d = batch_norm

        self.first_dconv = nn.Sequential(
            nn.Conv2D(24, 48, 1, bias_attr=False),
            BatchNorm2d(48),
            nn.ReLU6()
        )

        self.last_dconv = nn.Sequential(
            Conv2d(304, 256, kernel_size, 1, BatchNorm2d),
            Conv2d(256, 256, kernel_size, 1, BatchNorm2d)
        )

        self._init_weight()

    def forward(self, l, l_low):
        l_low = self.first_dconv(l_low)
        l = F.interpolate(l, size=l_low.shape[2:], mode='bilinear', align_corners=True)
        l = paddle.concat((l, l_low), axis=1)
        l = self.last_dconv(l)
        return l

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


# max-pooling indices-guided decoding
class IndexedDecoder(nn.Layer):
    def __init__(self, inp, oup, conv_operator='std_conv', kernel_size=5, batch_norm=SynchronizedBatchNorm2d):
        super(IndexedDecoder, self).__init__()
        Conv2d = conv[conv_operator]
        BatchNorm2d = batch_norm

        self.upsample = nn.MaxUnPool2D((2, 2), stride=2)
        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = Conv2d(inp, oup, kernel_size, 1, BatchNorm2d)

        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        l_encode = self.upsample(l_encode, indices) if indices is not None else l_encode
        l_encode = paddle.concat((l_encode, l_low), axis=1)    
        return self.dconv(l_encode)

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
    
    def visualize(self, x, indices=None):
        l = self.upsample(x, indices) if indices is not None else x
        l = l.mean(axis=1).squeeze()
        l = l.cpu().numpy()
        l = l / l.max() * 255.
        plt.figure()
        plt.imshow(l, cmap='viridis')
        plt.show()


class IndexedUpsamlping(nn.Layer):
    def __init__(self, inp, oup, conv_operator='std_conv', kernel_size=5, batch_norm=SynchronizedBatchNorm2d):
        super(IndexedUpsamlping, self).__init__()
        self.oup = oup

        Conv2d = conv[conv_operator]
        BatchNorm2d = batch_norm

        # inp, oup, kernel_size, stride, batch_norm
        self.dconv = Conv2d(inp, oup, kernel_size, 1, BatchNorm2d)

        self._init_weight()

    def forward(self, l_encode, l_low, indices=None):
        _, c, _, _ = l_encode.shape
        if indices is not None:
            l_encode = indices * F.interpolate(l_encode, size=l_low.shape[2:], mode='nearest')
        l_cat = paddle.concat((l_encode, l_low), axis=1)
        return self.dconv(l_cat)

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
    
    def visualize(self, x, indices=None):
        l = self.upsample(x, indices) if indices is not None else x
        l = l.mean(axis=1).squeeze()
        l = l.detach().cpu().numpy()
        l = l / l.max() * 255.
        plt.figure()
        plt.imshow(l, cmap='viridis')
        plt.axis('off')
        plt.show()
