import torch.nn as nn
from snot.models.ocean.modules import  Bottleneck, ResNet_plus2, Bottleneck_BIG_CI, ResNet

eps = 1e-5
# ---------------------
# For Ocean and Ocean+
# ---------------------
class ResNet50(nn.Module):
    def __init__(self, used_layers=[2, 3, 4], online=False):
        super(ResNet50, self).__init__()
        self.features = ResNet_plus2(Bottleneck, [3, 4, 6, 3], used_layers=used_layers, online=online)

    def forward(self, x, online=False):
        if not online:
            x_stages, x = self.features(x, online=online)
            return x_stages, x
        else:
            x = self.features(x, online=online)
            return x

# ---------------------
# For SiamDW
# ---------------------
class ResNet22W(nn.Module):
    """
    ResNet22W: double 3*3 layer (only) channels in residual blob
    """
    def __init__(self):
        super(ResNet22W, self).__init__()
        self.features = ResNet(Bottleneck_BIG_CI, [3, 4], [True, False], [False, True], firstchannels=64, channels=[64, 128])
        self.feature_size = 512

    def forward(self, x):
        x = self.features(x)

        return x
