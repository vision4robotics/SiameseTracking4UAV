# Parts of this code come from https://github.com/vision4robotics/SiamAPN
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from snot.core.config_adapn import cfg
from snot.models.backbone.alexnet import AlexNet_apn
from snot.models.adapn.utile import ADAPN,clsandloc_adapn
from snot.models.adapn.anchortarget import AnchorTarget3_adapn


class ModelBuilderADAPN(nn.Module):
    def __init__(self):
        super(ModelBuilderADAPN, self).__init__()

        self.backbone = AlexNet_apn().cuda()
        self.grader=ADAPN(cfg).cuda()
        self.new=clsandloc_adapn(cfg).cuda()
        self.fin2=AnchorTarget3_adapn() 

    def template(self, z):

        zf = self.backbone(z)
        self.zf=zf

    def track(self, x):

        xf = self.backbone(x)  
        xff,ress=self.grader(xf,self.zf)    

        self.ranchors=xff

        cls1,cls2,cls3,loc =self.new(xf,self.zf,ress)  

        return {
                'cls1': cls1,
                'cls2': cls2,
                'cls3': cls3,
                'loc': loc
               }
