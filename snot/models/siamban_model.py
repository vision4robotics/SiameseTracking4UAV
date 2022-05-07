# Parts of this code come from https://github.com/hqucv/siamban
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch.nn as nn

from snot.core.config_ban import cfg
from snot.models.backbone import get_backbone
from snot.models.head import get_ban_head
from snot.models.neck import get_ban_neck


class ModelBuilderBAN(nn.Module):
    def __init__(self):
        super(ModelBuilderBAN, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_ban_neck(cfg.ADJUST.TYPE,
                                     **cfg.ADJUST.KWARGS)

        # build ban head
        if cfg.BAN.BAN:
            self.head = get_ban_head(cfg.BAN.TYPE,
                                     **cfg.BAN.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):
        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)
        cls, loc = self.head(self.zf, xf)
        return {
                'cls': cls,
                'loc': loc
               }
