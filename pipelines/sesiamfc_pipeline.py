from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import yaml
from snot.models import sesiam_model as models
from snot.trackers.sesiam_fc_tracker import SESiamFCTracker
from snot.utils.bbox import get_axis_aligned_bbox
from snot.utils.model_load import load_pretrain


class SESiamFCPipeline():
    def __init__(self, args):
        super(SESiamFCPipeline, self).__init__()
        if not args.config:
            args.config = './experiments/SESiamFC/config.yaml'
        if not args.snapshot:
            args.snapshot = './experiments/SESiamFC/model.pth'

        with open(args.config) as f:
            tracker_config = yaml.load(f.read())
        self.net = models.__dict__[tracker_config['MODEL']](padding_mode='constant')
        self.net = load_pretrain(self.net, args.snapshot)
        self.net = self.net.eval().cuda()
        tracker_config = tracker_config['TRACKER']['VOT2017']
        self.tracker = SESiamFCTracker(self.net, **tracker_config)

    def init(self, img, gt_bbox):
        cx, cy, w, h = get_axis_aligned_bbox(np.array(gt_bbox))
        gt_bbox_ = [cx-(w-1)/2, cy-(h-1)/2, w, h]
        target_pos = np.array([cx, cy])
        target_sz = np.array([w, h])
        self.state = self.tracker.init(img, target_pos, target_sz)
        pred_bbox = gt_bbox_

        return pred_bbox
    
    def track(self, img):
        target_pos, target_sz = self.tracker.track(img)
        pred_bbox=np.array([target_pos[0]-target_sz[0]/2, target_pos[1]-target_sz[1]/2, target_sz[0], target_sz[1]])

        return pred_bbox

