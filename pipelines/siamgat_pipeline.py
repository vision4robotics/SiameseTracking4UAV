from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from snot.core.config_gat import cfg
from snot.models.siamgat_model import ModelBuilderGAT
from snot.trackers.siamgat_tracker import SiamGATTracker
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class SiamGATPipeline():
    def __init__(self, args):
        super(SiamGATPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamGAT/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SiamGAT/model.pth'

        cfg.merge_from_file(args.config)
        self.model = ModelBuilderGAT()
        self.model = load_pretrain(self.model, args.snapshot).cuda().eval()
        self.tracker = SiamGATTracker(self.model)

        cfg.TRACK.LR = 0.24
        cfg.TRACK.PENALTY_K = 0.04
        cfg.TRACK.WINDOW_INFLUENCE = 0.04

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        self.tracker.init(img, gt_bbox_)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        outputs = self.tracker.track(img)  
        pred_bbox = outputs['bbox']

        return pred_bbox

