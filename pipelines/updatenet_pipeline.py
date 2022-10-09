from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import torch

from snot.models.updatenet_siam_model import SiamRPNBIG
from snot.models.updatenet_upd_model import UpdateResNet
from snot.trackers.updatenet_tracker import SiamRPN_init, SiamRPN_track_upd
from snot.utils.bbox import get_axis_aligned_bbox


class UpdateNetPipeline():
    def __init__(self, args):
        super(UpdateNetPipeline, self).__init__()
        if not args.snapshot:
            args.snapshot = './experiments/UpdateNet/SiamRPNBIG.model'
        if not args.update:
            args.update = './experiments/UpdateNet/vot2018.pth.tar'

        self.net = SiamRPNBIG()
        self.net.load_state_dict(torch.load(args.snapshot))
        self.net.eval().cuda()
        self.updatenet = UpdateResNet()    
        self.updatenet.load_state_dict(torch.load(args.update)['state_dict'])
        self.updatenet.eval().cuda()

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        self.state = SiamRPN_init(img, target_pos, target_sz, self.net)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        self.state = SiamRPN_track_upd(self.state, img, self.updatenet) 
        target_pos=self.state['target_pos']
        target_sz=self.state['target_sz']
        pred_bbox=np.array([target_pos[0]-target_sz[0]/2, target_pos[1]-target_sz[1]/2, target_sz[0], target_sz[1]])

        return pred_bbox

