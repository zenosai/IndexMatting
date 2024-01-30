import paddle.nn as nn
from models.lib.modules import SynchronizedBatchNorm2d

def conv_bn(inp, oup, k=3, s=1, BatchNorm2d=SynchronizedBatchNorm2d):
    return nn.Sequential(
        nn.Conv2D(inp, oup, k, s, padding=k//2, bias_attr=False),
        BatchNorm2d(oup),
        nn.ReLU6()
    )    

def dep_sep_conv_bn(inp, oup, k=3, s=1, BatchNorm2d=SynchronizedBatchNorm2d):
    return nn.Sequential(
        nn.Conv2D(inp, inp, k, s, padding=k//2, groups=inp, bias_attr=False),
        BatchNorm2d(inp),
        nn.ReLU6(),
        nn.Conv2D(inp, oup, 1, 1, padding=0, bias_attr=False),
        BatchNorm2d(oup),
        nn.ReLU6()
    )

conv = {
    'std_conv': conv_bn,
    'dep_sep_conv': dep_sep_conv_bn
}