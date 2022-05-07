# Parts of this code come from https://github.com/isosnovik/SiamSE
from snot.models.se.siam_fc import SiamFC
from snot.models.se.feature_extractors import SEResNet22FeatureExtractor
from snot.models.se.connectors import ScaleHead


class SESiamFCResNet22(SiamFC):
    def __init__(self, padding_mode='circular', **kwargs):
        super().__init__(**kwargs)
        print('| using {} padding'.format(padding_mode))

        self.features = SEResNet22FeatureExtractor(scales=[0.9 * 1.4**i for i in range(3)],
                                                   pool=[False, True],
                                                   interscale=[True, False],
                                                   kernel_sizes=[9, 5, 5],
                                                   padding_mode=padding_mode)

        scales_head = [1 / 1.2, 1.0, 1.2]
        self.connect_model = ScaleHead(scales=scales_head, head='corr')
