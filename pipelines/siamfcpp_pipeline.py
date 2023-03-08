from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from snot.core.config_fcpp import cfg
from snot.core.config_fcpp import specify_task
from snot.models.fcpp.engine.builder import build as tester_builder
from snot.models.fcpp.model import builder as model_builder
from snot.models.fcpp.pipeline import builder as pipeline_builder
from snot.utils.bbox import get_axis_aligned_bbox


class SiamFCppPipeline():
    def __init__(self, args):
        super(SiamFCppPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SiamFC++/siamfcpp_googlenet.yaml'
        if not args.config:
            args.snapshot = './experiments/SiamFC++/siamfcpp-googlenet.pkl'

        cfg.merge_from_file(args.config)
        cfg['test']['track']['model']['task_model']['SiamTrack']['pretrain_model_path'] = args.snapshot
        task, task_cfg = specify_task(cfg['test'])
        task_cfg.freeze()
        self.model = model_builder.build("track", task_cfg.model)
        self.pipeline = pipeline_builder.build("track", task_cfg.pipeline, self.model)
        self.tester = tester_builder("track", task_cfg.tester, "tester", self.pipeline)[0]

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        self.tester.init(img, gt_bbox)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        pred_bbox = self.tester.track(img) 

        return pred_bbox

