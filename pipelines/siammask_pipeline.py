from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from snot.core.config import cfg
from snot.models.model_builder import ModelBuilder
from snot.trackers.tracker_builder import build_tracker
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class SiamMaskPipeline():
    def __init__(self, args):
        super(SiamMaskPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamMask_r50/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SiamMask_r50/model.pth'

        cfg.merge_from_file(args.config)
        self.model = ModelBuilder()
        self.model = load_pretrain(self.model, args.snapshot).cuda().eval()
        self.tracker = build_tracker(self.model)

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

