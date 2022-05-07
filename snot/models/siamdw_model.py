# Parts of this code come from https://github.com/researchmm/SiamDW and https://github.com/researchmm/TracKit
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from snot.models.dw.siamfc import SiamFC_
from snot.models.dw.siamrpn import SiamRPN_
from snot.models.dw.connect import Corr_Up, RPN_Up
from snot.models.dw.backbones import ResNet22, Incep22, ResNeXt22, ResNet22W


class SiamFCRes22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        self.connect_model = Corr_Up()


class SiamFCIncep22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCIncep22, self).__init__(**kwargs)
        self.features = Incep22()
        self.connect_model = Corr_Up()


class SiamFCNext22(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCNext22, self).__init__(**kwargs)
        self.features = ResNeXt22()
        self.connect_model = Corr_Up()


class SiamFCRes22W(SiamFC_):
    def __init__(self, **kwargs):
        super(SiamFCRes22W, self).__init__(**kwargs)
        self.features = ResNet22W()
        self.connect_model = Corr_Up()


class SiamRPNRes22(SiamRPN_):
    def __init__(self, **kwargs):
        super(SiamRPNRes22, self).__init__(**kwargs)
        self.features = ResNet22()
        inchannels = self.features.feature_size

        if self.cls_type == 'thinner': outchannels = 256
        elif self.cls_type == 'thicker': outchannels = 512
        else: raise ValueError('not implemented loss/cls type')

        self.connect_model = RPN_Up(anchor_nums=self.anchor_nums,
                                    inchannels=inchannels,
                                    outchannels=outchannels,
                                    cls_type = self.cls_type)
