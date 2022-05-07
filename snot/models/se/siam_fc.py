import torch
import torch.nn as nn
from torch.autograd import Variable


class SiamFC(nn.Module):
    def __init__(self):
        super(SiamFC, self).__init__()
        self.features = None
        self.connect_model = None
        self.zf = None  
        self.criterion = nn.BCEWithLogitsLoss()

    def template(self, z):
        self.zf = self.features(z)

    def _cls_loss(self, pred, label, select):
        if len(select.size()) == 0:
            return 0
        pred = torch.index_select(pred, 0, select)
        label = torch.index_select(label, 0, select)
        return self.criterion(pred, label)  

    def _weighted_BCE(self, pred, label):
        pred = pred.view(-1)
        label = label.view(-1)
        pos = Variable(label.data.eq(1).nonzero().squeeze()).cuda()
        neg = Variable(label.data.eq(0).nonzero().squeeze()).cuda()

        loss_pos = self._cls_loss(pred, label, pos)
        loss_neg = self._cls_loss(pred, label, neg)
        return loss_pos * 0.5 + loss_neg * 0.5

    def track(self, x):
        xf = self.features(x)
        score, activations = self.connect_model(self.zf, xf)
        return score, activations

    def forward(self, template, search, label=None):
        zf = self.features(template)
        xf = self.features(search)
        score, _ = self.connect_model(zf, xf)
        if self.training:
            return self._weighted_BCE(score, label)
        else:
            raise ValueError('forward is only used for training.')
