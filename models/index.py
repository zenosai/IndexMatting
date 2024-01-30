import paddle
import paddle.nn as nn
import paddle.nn.functional as F

class HolisticIndexBlock(nn.Layer):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=None):
        super(HolisticIndexBlock, self).__init__()

        BatchNorm2d = batch_norm
        
        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            self.indexnet = nn.Sequential(
                nn.Conv2D(inp, 2*inp, kernel_size=kernel_size, stride=2, padding=padding, bias_attr=False),
                BatchNorm2d(2*inp),
                nn.ReLU6(),
                nn.Conv2D(2*inp, 4, kernel_size=1, stride=1, padding=0, bias_attr=False)
            )
        else:
            self.indexnet = nn.Conv2D(inp, 4, kernel_size=kernel_size, stride=2, padding=padding, bias_attr=False)

    def forward(self, x):
        x = self.indexnet(x)
        
        y = F.sigmoid(x)
        z = F.softmax(y, axis=1)

        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseO2OIndexBlock(nn.Layer):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2D):
        super(DepthwiseO2OIndexBlock, self).__init__()

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        
    def _build_index_block(self, inp, use_nonlinear, use_context, BatchNorm2d):
        
        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2D(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias_attr=False),
                BatchNorm2d(inp),
                nn.ReLU6(),
                nn.Conv2D(inp, inp, kernel_size=1, stride=1, padding=0, groups=inp, bias_attr=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2D(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, groups=inp, bias_attr=False)
            )
        
    def forward(self, x):
        bs, c, h, w = x.shape

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = paddle.concat((x1, x2, x3, x4), axis=2)
        
        # normalization
        y = F.sigmoid(x)
        z = F.softmax(y, axis=2)
        # pixel shuffling
        y = y.reshape([bs, c*4, int(h/2), int(w/2)])
        z = z.reshape([bs, c*4, int(h/2), int(w/2)])
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de


class DepthwiseM2OIndexBlock(nn.Layer):
    def __init__(self, inp, use_nonlinear=False, use_context=False, batch_norm=nn.BatchNorm2D):
        super(DepthwiseM2OIndexBlock, self).__init__()
        self.use_nonlinear = use_nonlinear

        self.indexnet1 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet2 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet3 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        self.indexnet4 = self._build_index_block(inp, use_nonlinear, use_context, batch_norm)
        
    def _build_index_block(self, inp, use_nonlinear, use_context, BatchNorm2d):
        
        if use_context:
            kernel_size, padding = 4, 1
        else:
            kernel_size, padding = 2, 0

        if use_nonlinear:
            return nn.Sequential(
                nn.Conv2D(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias_attr=False),
                BatchNorm2d(inp),
                nn.ReLU6(),
                nn.Conv2D(inp, inp, kernel_size=1, stride=1, padding=0, bias_attr=False)
            )
        else:
            return nn.Sequential(
                nn.Conv2D(inp, inp, kernel_size=kernel_size, stride=2, padding=padding, bias_attr=False)
            )
        
    def forward(self, x):
        bs, c, h, w = x.shape

        x1 = self.indexnet1(x).unsqueeze(2)
        x2 = self.indexnet2(x).unsqueeze(2)
        x3 = self.indexnet3(x).unsqueeze(2)
        x4 = self.indexnet4(x).unsqueeze(2)

        x = paddle.concat((x1, x2, x3, x4), axis=2)
        
        # normalization
        y = F.sigmoid(x)
        z = F.softmax(y, axis=2)
        # pixel shuffling
        y = y.reshape([bs, c*4, int(h/2), int(w/2)])
        z = z.reshape([bs, c*4, int(h/2), int(w/2)])
        idx_en = F.pixel_shuffle(z, 2)
        idx_de = F.pixel_shuffle(y, 2)

        return idx_en, idx_de