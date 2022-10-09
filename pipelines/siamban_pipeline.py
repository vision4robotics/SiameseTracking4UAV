from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from snot.core.config_ban import cfg
from snot.models.siamban_model import ModelBuilderBAN
from snot.trackers.tracker_builder_ban import build_tracker_ban
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class SiamBANPipeline():
    def __init__(self, args):
        super(SiamBANPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamBAN/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SiamBAN/model.pth'

        cfg.merge_from_file(args.config)
        self.model = ModelBuilderBAN()
        self.model = load_pretrain(self.model, args.snapshot).cuda().eval()
        self.tracker = build_tracker_ban(self.model)

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

