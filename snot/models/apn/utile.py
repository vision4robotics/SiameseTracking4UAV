from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch as t


class APN(nn.Module):
    
    def __init__(self,cfg):
        super(APN, self).__init__()
        channels=cfg.TRAIN.apnchannel

        self.conv_shape = nn.Sequential(
                nn.Conv2d(channels, channels,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, 4,  kernel_size=3, stride=1,padding=1),
                )

        self.conv1=nn.Sequential(
                nn.Conv2d(channels, channels,  kernel_size=3, stride=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                )

        self.conv2=nn.Sequential(
                nn.Conv2d(channels, channels,  kernel_size=3, stride=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                )

        for modules in [self.conv_shape,self.conv1,self.conv2]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)

    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out
    
    def forward(self,x,z):
        x=self.conv1(x)
        z=self.conv2(z)
        res=self.xcorr_depthwise(x,z)
        shape_pred=self.conv_shape(res)

        return shape_pred,res


class clsandloc_apn(nn.Module):

    def __init__(self,cfg):
        super(clsandloc_apn, self).__init__()
        channel=cfg.TRAIN.clsandlocchannel
        
        self.conv1=nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.conv2=nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace=True),
                )

        self.convloc = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, 4,  kernel_size=3, stride=1,padding=1),
                )

        self.convcls = nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                )

        self.conv_offset = nn.Sequential(
                nn.Conv2d(cfg.TRAIN.apnchannel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1),
                )

        self.add=nn.ConvTranspose2d(channel * 2, channel, 3, 1)

        self.resize= nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.GroupNorm(cfg.TRAIN.groupchannel,channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1),
                )

        self.relu=nn.ReLU(inplace=True)

        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls3=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)

        for modules in [self.convloc, self.convcls,
                       self.resize, self.cls1,self.add,self.conv_offset,
                        self.conv1,self.conv2,
                        self.cls2,self.cls3]:
            for l in modules.modules():
               if isinstance(l, nn.Conv2d):
                    t.nn.init.normal_(l.weight, std=0.01)
                    t.nn.init.constant_(l.bias, 0)

    def xcorr_depthwise(self,x, kernel):
        """depthwise cross correlation
        """
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch*channel, x.size(2), x.size(3))
        kernel = kernel.view(batch*channel, 1, kernel.size(2), kernel.size(3))
        out = F.conv2d(x, kernel, groups=batch*channel)
        out = out.view(batch, channel, out.size(2), out.size(3))
        return out

    def forward(self, x,z,ress):
        x=self.conv1(x)
        z=self.conv2(z)
        res=self.xcorr_depthwise(x,z)
        res=self.resize(res)
        ress=self.conv_offset(ress)
        res=self.add(self.relu(t.cat((res,ress),1))) 

        cls=self.convcls(res)
        cls1=self.cls1(cls)
        cls2=self.cls2(cls)
        cls3=self.cls3(cls)

        loc=self.convloc(res)

        return cls1,cls2,cls3,loc
