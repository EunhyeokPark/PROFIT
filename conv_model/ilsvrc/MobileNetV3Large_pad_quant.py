from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

_BN_MOMENTUM = 1 - 0.9997


def ch_8x(ch):
    return int(((ch + 7) // 8) * 8)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SEModule(nn.Module):
    def __init__(self, ops, channels, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        se_channel = ch_8x(channels / reduction)

        self.fc = ops.Sequential(
            ops.Sym(),
            ops.Linear(channels, se_channel, bias=True),
            ops.ReLU(inplace=True),
            ops.Linear(se_channel, channels, bias=True),
            h_sigmoid()
        )

    def forward(self, x):
        x_avg = self.avg_pool(x).view(x.shape[0], -1)
        return x * self.fc(x_avg).view(x.shape[0], -1, 1, 1)


def get_nonlinear(nonlinear):
    if nonlinear == "RE":
        return nn.ReLU(inplace=True)
    elif nonlinear == "HS":
        return h_swish()
    else:
        raise LookupError("Unknown nonlinear")


def get_quant(ops, nonlinear):
    if nonlinear == "RE":
        return ops.ReLU(inplace=True)
    elif nonlinear == "HS":
        return ops.HSwish()
    else:
        raise LookupError("Unknown nonlinear")


class Conv2dPad(nn.Conv2d):
    def __init__(self, mode, *args, **kargs):
        super(Conv2dPad, self).__init__(*args, **kargs)
        self.mode = mode

    def forward(self, input):
        if self.mode == "HS":
            input = F.pad(input, self.padding + self.padding, value=-0.375)
        elif self.mode == "RE":
            input = F.pad(input, self.padding + self.padding, value=0)
        else:
            raise LookupError("Unknown nonlinear")
        
        return F.conv2d(input, self.weight, self.bias, self.stride,
                0, self.dilation, self.groups)


class MobileNetV3Block(nn.Module):
    def __init__(self, ops, scale, in_channels, out_channels, kernel_size, expansion, squeeze_excite, nonlinear, stride):
        super(MobileNetV3Block, self).__init__()

        self.identity = stride == 1 and in_channels == out_channels
        in_channels = ch_8x(in_channels * scale)
        out_channels = ch_8x(out_channels * scale)
        expansion = ch_8x(expansion * scale)

        if in_channels == expansion:
            self.conv = ops.Sequential(
                # dw
                ops.HSwish(False),
                ops.Conv2dPad("HS", in_channels, in_channels, kernel_size, stride, (kernel_size - 1) // 2, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels, momentum=_BN_MOMENTUM),
                # Squeeze-and-Excite
                SEModule(ops, in_channels) if squeeze_excite else nn.Sequential(),
                get_quant(ops, nonlinear),
                # pw-linear
                ops.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
            )
        else:
            self.conv = ops.Sequential(
                # pw
                ops.Sym(),
                ops.Conv2d(in_channels, expansion, 1, 1, 0, bias=False),
                nn.BatchNorm2d(expansion, momentum=_BN_MOMENTUM),
                get_quant(ops, nonlinear),
                # dw
                ops.Conv2dPad(nonlinear, expansion, expansion, kernel_size, stride, (kernel_size - 1) // 2, groups=expansion, bias=False),
                nn.BatchNorm2d(expansion, momentum=_BN_MOMENTUM),
                # Squeeze-and-Excite
                SEModule(ops, expansion) if squeeze_excite else nn.Sequential(),
                get_quant(ops, nonlinear),
                # pw-linear
                ops.Conv2d(expansion, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


MV3BlockConfig = namedtuple("MV3BlockConfig", ["out_channels", "kernel_size", "expansion", "squeeze_excite", "nonlinear", "stride"])

MV3LargeBlocks = [
    MV3BlockConfig(16, 3, 16, False, "RE", 1),
    MV3BlockConfig(24, 3, 64, False, "RE", 2),
    MV3BlockConfig(24, 3, 72, False, "RE", 1),

    MV3BlockConfig(40, 5, 72, True, "RE", 2),
    MV3BlockConfig(40, 5, 120, True, "RE", 1),
    MV3BlockConfig(40, 5, 120, True, "RE", 1),

    MV3BlockConfig(80, 3, 240, False, "HS", 2),
    MV3BlockConfig(80, 3, 200, False, "HS", 1),
    MV3BlockConfig(80, 3, 184, False, "HS", 1),
    MV3BlockConfig(80, 3, 184, False, "HS", 1),

    MV3BlockConfig(112, 3, 480, True, "HS", 1),
    MV3BlockConfig(112, 3, 672, True, "HS", 1),

    MV3BlockConfig(160, 5, 672, True, "HS", 2),
    MV3BlockConfig(160, 5, 960, True, "HS", 1),
    MV3BlockConfig(160, 5, 960, True, "HS", 1),
]


def conv_3x3_bn(ops, in_channels, out_channels, stride, nonlinear):
    return nn.Sequential(
        ops.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
        nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
        get_nonlinear(nonlinear)
    )


def conv_1x1_bn(ops, in_channels, out_channels, stride, nonlinear):
    return nn.Sequential(
        ops.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM),
        get_nonlinear(nonlinear)
    )


class MobileNetV3Large(nn.Module):
    def __init__(self, ops, scale=1., num_classes=1000):
        super(MobileNetV3Large, self).__init__()

        # building first layer
        layers = [conv_3x3_bn(ops, 3, 16, 2, "HS")]
        in_channels = 16

        # building inverted residual blocks
        for block in MV3LargeBlocks:
            layers.append(MobileNetV3Block(ops, scale, in_channels, *block))
            in_channels = block.out_channels

        in_ch = ch_8x(in_channels * scale)
        ch = ch_8x(960 * scale)
        ch2 = ch_8x(1280 * scale)
        layers.append(ops.Sym())
        layers.append(conv_1x1_bn(ops, in_ch, ch, 1, "HS"))
        self.features = ops.Sequential(*layers)

        # building fully connected blocks
        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = ops.Sequential(
            ops.HSwish(False),
            ops.Linear(ch, ch2, bias=True),
            get_quant(ops, "HS"),
            nn.Dropout(p=0.2),
            #nn.Dropout(),
            ops.Linear(ch2, num_classes, bias=True)
        )

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out",
                                         nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = x.view(x.shape[0], -1)
        x = self.classifier(x)
        return x
