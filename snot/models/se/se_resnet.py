import torch.nn as nn
import torch.nn.functional as F

from snot.models.se.sesn.ses_conv import SESConv_Z2_H, SESConv_H_H_1x1, SESConv_H_H, ses_max_projection


def center_crop(x, crop):
    return F.pad(x, [-crop, -crop, -crop, -crop, 0, 0])


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, last_relu, stride=1, downsample=None,
                 scales=[1.0], kernel_size=7, pool=False,
                 interscale=False, padding_mode='constant'):
        super(Bottleneck, self).__init__()
        self.pool = pool
        padding = kernel_size // 2

        if pool:
            self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)

            self.bn1 = nn.BatchNorm2d(planes)
            self.conv2 = SESConv_Z2_H(planes, planes, kernel_size=kernel_size,
                                      effective_size=3, scales=scales,
                                      stride=stride, padding=padding,
                                      bias=False, padding_mode=padding_mode)
        else:
            self.conv1 = SESConv_H_H_1x1(inplanes, planes, 2 if interscale else 1, bias=False)

            self.bn1 = nn.BatchNorm3d(planes)
            self.conv2 = SESConv_H_H(planes, planes,
                                     scale_size=1, kernel_size=kernel_size,
                                     effective_size=3, scales=scales,
                                     stride=stride, padding=padding,
                                     bias=False, padding_mode=padding_mode)

        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = SESConv_H_H_1x1(planes, planes * self.expansion,
                                     num_scales=len(scales), bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def forward(self, x):
        residual = x

        if self.pool:
            x = ses_max_projection(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(residual)
        out = center_crop(out + residual, 1)

        if self.last_relu:
            out = self.relu(out)

        return out


class SEResNet(nn.Module):

    def __init__(self, layers, last_relus, s2p_flags, firstchannels=64,
                 channels=[64, 128], scales=[1.0], pool=[False, True],
                 interscale=[False, False], kernel_sizes=[11, 7, 7], padding_mode='constant'):

        super(SEResNet, self).__init__()

        self.inplanes = firstchannels
        self.stage_len = len(layers)
        padding = kernel_sizes[0] // 2

        self.conv1 = SESConv_Z2_H(3, firstchannels, kernel_size=kernel_sizes[0],
                                  effective_size=7, scales=scales,
                                  padding=padding, stride=2, bias=False,
                                  padding_mode=padding_mode)

        self.bn1 = nn.BatchNorm3d(firstchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.layer1 = self._make_layer(channels[0], layers[0],
                                       stride2pool=s2p_flags[0], last_relu=last_relus[0], scales=scales,
                                       kernel_size=kernel_sizes[1], pool=pool[0],
                                       interscale=interscale[0], padding_mode=padding_mode)

        self.layer2 = self._make_layer(channels[1], layers[1],
                                       stride2pool=s2p_flags[1], last_relu=last_relus[1],
                                       scales=scales, kernel_size=kernel_sizes[2], pool=pool[1],
                                       interscale=interscale[1], padding_mode=padding_mode)

        for m in self.modules():
            if isinstance(m, (SESConv_H_H, SESConv_Z2_H, SESConv_H_H_1x1)):
                nelement = m.weight.nelement()
                n = nelement / m.in_channels
                m.weight.data.normal_(0, (2 / n)**0.5)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, planes, blocks, last_relu, stride=1, stride2pool=False,
                    scales=[1.0], kernel_size=7, pool=False,
                    interscale=False, padding_mode='constant'):

        downsample = None
        if stride != 1 or self.inplanes != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                SESConv_H_H_1x1(self.inplanes, planes * Bottleneck.expansion,
                                num_scales=len(scales), stride=stride, bias=False),
                nn.BatchNorm3d(planes * Bottleneck.expansion),
            )

        layers = []
        layers.append(Bottleneck(self.inplanes, planes, last_relu=True, stride=stride, downsample=downsample,
                                 scales=scales, kernel_size=kernel_size, pool=pool,
                                 interscale=interscale, padding_mode=padding_mode))
        if stride2pool:
            layers.append(self.maxpool)
        self.inplanes = planes * Bottleneck.expansion
        for i in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes,
                                     last_relu=last_relu if (i == blocks - 1) else True,
                                     scales=scales,
                                     kernel_size=kernel_size,
                                     pool=False,
                                     interscale=False,
                                     padding_mode=padding_mode))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = center_crop(x, 2)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return x
