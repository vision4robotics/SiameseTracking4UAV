from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from snot.models.neck.neck import AdjustLayer, AdjustAllLayer, BAN_AdjustLayer, BAN_AdjustAllLayer

NECKS = {
         'AdjustLayer': AdjustLayer,
         'AdjustAllLayer': AdjustAllLayer
        }

def get_neck(name, **kwargs):
    return NECKS[name](**kwargs)

BAN_NECKS = {
         'AdjustLayer': BAN_AdjustLayer,
         'AdjustAllLayer': BAN_AdjustAllLayer
        }

def get_ban_neck(name, **kwargs):
    return BAN_NECKS[name](**kwargs)
