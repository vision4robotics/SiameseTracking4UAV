# Parts of this code come from https://github.com/ohhhyeahhh/SiamCAR
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn

from snot.core.config_car import cfg
from snot.models.backbone import get_backbone
from snot.models.head.car import CAR
from snot.models.neck import get_neck
from snot.core.xcorr import xcorr_depthwise

class ModelBuilderCAR(nn.Module):
    def __init__(self):
        super(ModelBuilderCAR, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build car head
        self.car_head = CAR(cfg, 256)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        features = self.xcorr_depthwise(xf[0],self.zf[0])
        for i in range(len(xf)-1):
            features_new = self.xcorr_depthwise(xf[i+1],self.zf[i+1])
            features = torch.cat([features,features_new],1)
        features = self.down(features)

        cls, loc, cen = self.car_head(features)
        return {
                'cls': cls,
                'loc': loc,
                'cen': cen
               }
