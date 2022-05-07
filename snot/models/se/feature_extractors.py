import torch
import torch.nn as nn

from snot.models.se.se_resnet import SEResNet

eps = 1e-5


class SEResNet22FeatureExtractor(nn.Module):

    def __init__(self, scales=[1.0], pool=[False, True], interscale=[False, False],
                 kernel_sizes=[11, 7, 7], padding_mode='constant'):

        super().__init__()
        self.features = SEResNet(layers=[3, 4],
                                 last_relus=[True, False],
                                 s2p_flags=[False, True],
                                 firstchannels=64,
                                 channels=[64, 128],
                                 scales=scales,
                                 pool=pool,
                                 interscale=interscale,
                                 kernel_sizes=kernel_sizes,
                                 padding_mode=padding_mode)

        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]))
        self.feature_size = 512
        self.train_num = 0
        self.unfix(0.0)

    def forward(self, x):
        x = x / 255.0
        x = x - self.mean.view(1, -1, 1, 1)
        x = x / self.std.view(1, -1, 1, 1)
        x = self.features(x)
        return x

    def unfix(self, ratio):
        if abs(ratio - 0.0) < eps:
            self.train_num = 2
            self.unlock()
            return True
        elif abs(ratio - 0.1) < eps:
            self.train_num = 3
            self.unlock()
            return True
        elif abs(ratio - 0.2) < eps:
            self.train_num = 4
            self.unlock()
            return True
        elif abs(ratio - 0.3) < eps:
            self.train_num = 6
            self.unlock()
            return True
        elif abs(ratio - 0.5) < eps:
            self.train_num = 7
            self.unlock()
            return True
        elif abs(ratio - 0.6) < eps:
            self.train_num = 8
            self.unlock()
            return True
        elif abs(ratio - 0.7) < eps:
            self.train_num = 10
            self.unlock()
            return True

        return False

    def unlock(self):
        for p in self.parameters():
            p.requires_grad = False

        for i in range(1, self.train_num):
            if i <= 5:
                m = self.features.layer2[-i]
            elif i <= 8:
                m = self.features.layer1[-(i - 5)]
            else:
                m = self.features

            for p in m.parameters():
                p.requires_grad = True
        self.eval()
        self.train()

    def train(self, mode=True):
        self.training = mode
        if mode == False:
            super().train(False)
        else:
            for i in range(self.train_num):
                if i <= 5:
                    m = self.features.layer2[-i]
                elif i <= 8:
                    m = self.features.layer1[-(i - 5)]
                else:
                    m = self.features
                m.train(mode)

        return self
