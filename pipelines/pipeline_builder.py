from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


from snot.pipelines.dasiamrpn_pipeline import DaSiamRPNPipeline
from snot.pipelines.lighttrack_pipeline import LightTrackPipeline
from snot.pipelines.ocean_pipeline import OceanPipeline
from snot.pipelines.sesiamfc_pipeline import SESiamFCPipeline
from snot.pipelines.siamapn_pipeline import SiamAPNPipeline
from snot.pipelines.siamapnpp_pipeline import SiamAPNppPipeline
from snot.pipelines.siamban_pipeline import SiamBANPipeline
from snot.pipelines.siamcar_pipeline import SiamCARPipeline
from snot.pipelines.siamdw_fc_pipeline import SiamDWFCPipeline
from snot.pipelines.siamfcpp_pipeline import SiamFCppPipeline
from snot.pipelines.siamgat_pipeline import SiamGATPipeline
from snot.pipelines.siammask_pipeline import SiamMaskPipeline
from snot.pipelines.siamdw_rpn_pipeline import SiamDWRPNPipeline
from snot.pipelines.siamrpn_pipeline import SiamRPNppPipeline
from snot.pipelines.updatenet_pipeline import UpdateNetPipeline

TRACKS = {
          'DaSiamRPN': DaSiamRPNPipeline,
          'LightTrack': LightTrackPipeline,
          'Ocean': OceanPipeline,
          'SESiamFC': SESiamFCPipeline,
          'SiamAPN': SiamAPNPipeline,
          'SiamAPN++': SiamAPNppPipeline,
          'SiamBAN': SiamBANPipeline,
          'SiamCAR': SiamCARPipeline,
          'SiamFC+': SiamDWFCPipeline,
          'SiamFC++': SiamFCppPipeline,
          'SiamGAT': SiamGATPipeline,
          'SiamMask': SiamMaskPipeline,
          'SiamRPN+': SiamDWRPNPipeline,
          'SiamRPN++': SiamRPNppPipeline,
          'UpdateNet': UpdateNetPipeline
         }

def build_pipeline(args):
    return TRACKS[args.trackername.split('_')[0]](args)