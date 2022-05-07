# Parts of this code come from https://github.com/researchmm/TracKit
from snot.models.ocean.ocean import Ocean_
from snot.models.ocean.connect import box_tower, AdjustLayer
from snot.models.ocean.backbones import ResNet50


class Ocean(Ocean_):
    def __init__(self, align=False, online=False):
        super(Ocean, self).__init__()
        self.features = ResNet50(used_layers=[3], online=online)   # in param
        self.neck = AdjustLayer(in_channels=1024, out_channels=256)
        self.connect_model = box_tower(inchannels=256, outchannels=256, towernum=4)
        self.align_head = None
