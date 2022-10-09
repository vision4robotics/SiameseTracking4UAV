from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from snot.core.config_car import cfg
from snot.models.siamcar_model import ModelBuilderCAR
from snot.trackers.siamcar_tracker import SiamCARTracker
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class SiamCARPipeline():
    def __init__(self, args):
        super(SiamCARPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamCAR/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SiamCAR/model.pth'

        cfg.merge_from_file(args.config)
        self.model = ModelBuilderCAR()
        self.model = load_pretrain(self.model, args.snapshot).cuda().eval()
        self.tracker = SiamCARTracker(self.model, cfg.TRACK)

        self.hp = {'lr': 0.4, 'penalty_k': 0.2, 'window_lr': 0.3}

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        self.tracker.init(img, gt_bbox_)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        outputs = self.tracker.track(img, self.hp)  
        pred_bbox = outputs['bbox']

        return pred_bbox

