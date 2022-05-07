# Parts of this code come from https://github.com/hqucv/siamban
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from snot.core.config_ban import cfg
from snot.trackers.siamban_tracker import SiamBANTracker

TRACKS = {
          'SiamBANTracker': SiamBANTracker
         }


def build_tracker_ban(model):
    return TRACKS[cfg.TRACK.TYPE](model)
