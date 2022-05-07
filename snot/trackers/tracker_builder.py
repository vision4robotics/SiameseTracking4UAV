# Parts of this code come from https://github.com/STVIR/pysot
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from snot.core.config import cfg
from snot.trackers.siamrpn_tracker import SiamRPNTracker
from snot.trackers.siammask_tracker import SiamMaskTracker
from snot.trackers.siamrpnlt_tracker import SiamRPNLTTracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)
