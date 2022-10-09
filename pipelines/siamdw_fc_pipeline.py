from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from easydict import EasyDict as edict
from snot.models import siamdw_model as models
from snot.trackers.siamdw_fc_tracker import SiamFC
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class SiamDWFCPipeline():
    def __init__(self, args):
        super(SiamDWFCPipeline, self).__init__()
        if not args.arch:
            args.arch = 'SiamFCRes22'
        if not args.snapshot:
            args.snapshot = './experiments/SiamDW_FCRes22/model.pth'

        self.net = models.__dict__[args.arch]()
        self.net = load_pretrain(self.net, args.snapshot)
        self.net.eval()                 
        self.net = self.net.cuda()
        self.info = edict()
        self.info.arch = args.arch
        self.info.dataset = args.dataset
        self.info.epoch_test = args.epoch_test
        self.tracker = SiamFC(self.info)

        self.hp = {'SiamFCIncep22':{'scale_step': 1.1679, 'scale_lr': 0.6782, 'scale_penalty': 0.9285, 'w_influence': 0.2566},
                'SiamFCNext22':{'scale_step': 1.1531, 'scale_lr': 0.5706, 'scale_penalty': 0.9489, 'w_influence': 0.2581, 'instance_size':255},
                'SiamFCRes22':{'scale_step': 1.1466, 'scale_lr': 0.2061, 'scale_penalty': 0.9994, 'w_influence': 0.3242}}

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

