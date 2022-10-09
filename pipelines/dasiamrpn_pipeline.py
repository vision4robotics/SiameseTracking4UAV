from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from snot.models.dasiamrpn_model import SiamRPNBIG
from snot.trackers.dasiamrpn_tracker import SiamRPN_init, SiamRPN_track
from snot.utils.bbox import get_axis_aligned_bbox


class DaSiamRPNPipeline():
    def __init__(self, args):
        super(DaSiamRPNPipeline, self).__init__()
        if not args.snapshot:
            args.snapshot = './experiments/DaSiamRPN/SiamRPNBIG.model'

        self.net = SiamRPNBIG()
        self.net.load_state_dict(torch.load(args.snapshot))
        self.net.eval().cuda()

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        self.state = SiamRPN_init(img, target_pos, target_sz, self.net)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        self.state = SiamRPN_track(self.state, img) 
        target_pos=self.state['target_pos']
        target_sz=self.state['target_sz']
        pred_bbox=np.array([target_pos[0]-target_sz[0]/2, target_pos[1]-target_sz[1]/2, target_sz[0], target_sz[1]])

        return pred_bbox

