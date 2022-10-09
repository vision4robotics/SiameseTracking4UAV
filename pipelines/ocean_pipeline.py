from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from easydict import EasyDict as edict
from snot.models import ocean_model as models
from snot.trackers.ocean_tracker import Ocean
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class OceanPipeline():
    def __init__(self, args):
        super(OceanPipeline, self).__init__()
        if not args.arch:
            args.arch = 'Ocean'
        if not args.snapshot:
            args.snapshot = './experiments/Ocean/model.pth'

        self.net = models.__dict__[args.arch](align=args.align, online=args.online)
        self.net = load_pretrain(self.net, args.snapshot)
        self.net.eval()
        self.net = self.net.cuda()
        self.info = edict()
        self.info.arch = args.arch
        self.info.TRT = 'TRT' in args.arch
        self.info.dataset = args.dataset
        self.info.align = args.align
        self.info.online = args.online
        self.info.epoch_test = args.epoch_test
        self.tracker = Ocean(self.info)

        self.hp = {'Ocean':{'penalty_k': 0.08, 'lr': 0.305, 'window_influence': 0.44, 'small_sz': 127, 'big_sz': 287}}

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        self.state = self.tracker.init(img, target_pos, target_sz, self.net, self.hp[self.info.arch])
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        self.state = self.tracker.track(self.state, img)
        target_pos=self.state['target_pos']
        target_sz=self.state['target_sz']
        pred_bbox=np.array([target_pos[0]-target_sz[0]/2, target_pos[1]-target_sz[1]/2, target_sz[0], target_sz[1]])

        return pred_bbox

