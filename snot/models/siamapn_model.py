# Parts of this code come from https://github.com/vision4robotics/SiamAPN
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from snot.core.config_apn import cfg
from snot.models.backbone.alexnet import AlexNet_apn
from snot.models.apn.utile import APN,clsandloc_apn
from snot.models.apn.anchortarget import AnchorTarget3_apn


class ModelBuilderAPN(nn.Module):
    def __init__(self):
        super(ModelBuilderAPN, self).__init__()

        self.backbone = AlexNet_apn().cuda()
        self.grader=APN(cfg).cuda()
        self.new=clsandloc_apn(cfg).cuda()
        self.fin2=AnchorTarget3_apn()         

    def template(self, z):

        zf1,zf = self.backbone(z)
        self.zf=zf
        self.zf1=zf1

    def track(self, x):

        xf1,xf = self.backbone(x)  
        xff,ress=self.grader(xf1,self.zf1)    

        self.ranchors=xff

        cls1,cls2,cls3,loc =self.new(xf,self.zf,ress)  

        return {
                'cls1': cls1,
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }
