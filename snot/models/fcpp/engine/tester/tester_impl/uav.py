# -*- coding: utf-8 -*
import copy

import torch

from ..tester_base import TRACK_TESTERS, TesterBase
from .utils.benchmark_helper import PipelineTracker


@TRACK_TESTERS.register
class UAVTester(TesterBase):
    r"""UAV tester
    """
    extra_hyper_params = dict(
        device_num=1,
        data_root="",
        subsets=["UAV123"],  # (UAV123|UAV20L)
    )

    def __init__(self, *args, **kwargs):
        super(UAVTester, self).__init__(*args, **kwargs)
        # self._experiment = None

    def update_params(self):
        # set device state
        num_gpu = self._hyper_params["device_num"]
        if num_gpu > 0:
            all_devs = [torch.device("cuda:%d" % i) for i in range(num_gpu)]
        else:
            all_devs = [torch.device("cpu")]
        self._state["all_devs"] = all_devs
    
    def init(self, img, gt_bbox):
        tracker_name = self._hyper_params["exp_name"]
        for subset in self._hyper_params["subsets"]:
            self._pipeline.set_device(1)
            self.pipeline_tracker = PipelineTracker(tracker_name, self._pipeline)
            self.pipeline_tracker.init(img, gt_bbox)
    
    def track(self, img):
        boxes = self.pipeline_tracker.update(img)
        return boxes

UAVTester.default_hyper_params = copy.deepcopy(UAVTester.default_hyper_params)
UAVTester.default_hyper_params.update(UAVTester.extra_hyper_params)
