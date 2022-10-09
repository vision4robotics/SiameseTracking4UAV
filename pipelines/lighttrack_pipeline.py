from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from easydict import EasyDict as edict
from snot.models import lighttrack_model as models
from snot.trackers.lighttrack_tracker import Lighttrack
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class LightTrackPipeline():
    def __init__(self, args):
        super(LightTrackPipeline, self).__init__()
        if not args.arch:
            args.arch = 'LightTrackM_Subnet'
        if not args.snapshot:
            args.snapshot = './experiments/LightTrack/model.pth'
        if not args.path_name:
            args.path_name = 'back_04502514044521042540+cls_211000022+reg_100000111_ops_32'

        self.info = edict()
        self.info.arch = args.arch
        self.info.dataset = args.dataset
        self.info.epoch_test = args.epoch_test
        self.info.stride = args.stride
        self.net = models.__dict__[args.arch](args.path_name, stride=self.info.stride)
        self.net = load_pretrain(self.net, args.snapshot)
        self.net.eval()
        self.net = self.net.cuda()
        self.tracker = Lighttrack(self.info, even=args.even)

        self.hp = {'LightTrackM_Subnet':{'penalty_k': 0.007, 'lr': 0.616, 'window_influence': 0.225, 'small_sz': 256, 'big_sz': 288, 'ratio': 1}}

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

