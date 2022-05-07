from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn
import torch.nn.functional as F
import torch as t


class ADAPN(nn.Module):
    
    def __init__(self,cfg):
        super(ADAPN, self).__init__()
        channels=cfg.TRAIN.apnchannel
        channelr=cfg.TRAIN.clsandlocchannel

        self.conv_shape = nn.Sequential(
                nn.Conv2d(channelr, channelr,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channelr),  
                nn.ReLU(inplace=True),
                )

        self.anchor=nn.Conv2d(channelr, 4,  kernel_size=3, stride=1,padding=1)

        self.conv3=nn.Sequential(
                nn.Conv2d(channels, channelr,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channelr),  
                nn.ReLU(inplace=True),
                )

        self.conv5=nn.Sequential(
                nn.Conv2d(channelr, channelr,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channelr),  
                nn.ReLU(inplace=True),
                )

        self.conv6=nn.Sequential(
                nn.Conv2d(channelr, channelr,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channelr),   
                nn.ReLU(inplace=True),
                )

        self.adcat=adcat(256)

        for modules in [self.conv3,self.conv5,self.conv6,self.conv_shape]:
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

        res2=self.conv3(self.xcorr_depthwise(x[0],z[0]))
        ress=self.xcorr_depthwise(self.conv5(x[1]),self.conv6(z[1]))
        ress=self.adcat(ress,res2)

        res=self.conv_shape(ress)
        shape_pred=self.anchor(res)

        return shape_pred,ress


class clsandloc_adapn(nn.Module):

    def __init__(self,cfg):
        super(clsandloc_adapn, self).__init__()
        channel=cfg.TRAIN.clsandlocchannel
        
        self.conv1=nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel), 
                nn.ReLU(inplace=True),
                )

        self.conv2=nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(channel), 
                nn.ReLU(inplace=True),
                )

        self.conv4=nn.Sequential(
                nn.Conv2d(channel, channel,  kernel_size=3, stride=1,padding=1),
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
        
        self.channel=selfchannelbranch(channel)
        self.point=selfpointbranch(channel)
        
        self.cls1=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls2=nn.Conv2d(channel, 2,  kernel_size=3, stride=1,padding=1)
        self.cls3=nn.Conv2d(channel, 1,  kernel_size=3, stride=1,padding=1)
        self.adcat=adcat(channel)

        for modules in [self.convloc, self.convcls, self.cls1,
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
        res=self.xcorr_depthwise(self.conv1(x[1]),self.conv2(z[1]))
        
        point=self.point(res)
        channel=self.conv4(self.channel(point))
        
        res=self.adcat(channel,ress)

        cls=self.convcls(res)
        cls1=self.cls1(cls)
        cls2=self.cls2(cls)
        cls3=self.cls3(cls)

        loc=self.convloc(res)

        return cls1,cls2,cls3,loc


class selfpointbranch(nn.Module):

    def __init__(self, in_dim):
        super(selfpointbranch, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(t.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = t.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = t.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

    
class selfchannelbranch(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(selfchannelbranch, self).__init__()
        self.chanel_in = in_dim

        self.conv1=nn.Sequential(
                 nn.Conv2d(in_dim, 256,  kernel_size=7, stride=1),
                 nn.BatchNorm2d(256),  
                 nn.ReLU(inplace=True),
                 )
        
        self.gamma2 = nn.Parameter(t.zeros(1).cuda())

        self.softmax  = nn.Softmax(dim=-1)
        
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
        
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        out2=self.sigmoid(max_out+avg_out)  
        
        out = x+self.gamma2*out2*x

        return out

    
class adcat(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(adcat, self).__init__()
        self.chanel_in = in_dim

        self.add=nn.ConvTranspose2d(self.chanel_in*2, self.chanel_in,  kernel_size=1, stride=1)
                
        self.fc1   = nn.Conv2d(in_dim, in_dim // 6, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_dim // 6, in_dim, 1, bias=False)
        
        self.gamma1 = nn.Parameter(t.zeros(1).cuda())
        self.gamma3 = nn.Parameter(t.zeros(1).cuda())
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self,x,z):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        w=self.sigmoid(self.fc2(self.relu1(self.fc1(self.avg_pool(z)))))
        c2=self.relu1(self.add(t.cat((x,z),1)))
        out=x+self.gamma1*c2+self.gamma3*w*x

        return out
