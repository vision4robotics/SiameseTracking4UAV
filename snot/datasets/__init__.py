from .uav10fps import UAV10Dataset
from .uav20l import UAV20Dataset
from .dtb import DTBDataset
from .uavdt import UAVDTDataset
from .visdrone import VISDRONEDDataset
from .v4r import V4RDataset
from .uavdark import UAVDARKDataset
from .darktrack import DARKTRACKDataset


datapath = {
            'UAV10':'/Dataset/UAV123_10fps',
            'UAV20':'/Dataset/UAV123_20L',
            'DTB70':'/Dataset/DTB70',
            'UAVDT':'/Dataset/UAVDT',
            'VISDRONED2018':'/Dataset/VisDrone-SOT2018-test',
            'VISDRONED2019':'/Dataset/VisDrone-SOT2019-test',
            'VISDRONED2020':'/Dataset/VisDrone-SOT2020-test',
            'UAVTrack112':'/Dataset/UAVTrack112',
            'UAVDark135':'/Dataset/UAVDark135',
            'DarkTrack2021':'/Dataset/DarkTrack2021',
            'NAT2021':'/Dataset/NAT2021'
            }

class DatasetFactory(object):
    @staticmethod
    def create_dataset(**kwargs):

        assert 'name' in kwargs, "should provide dataset name"
        name = kwargs['name']
        if 'UAV10' in name:
            dataset = UAV10Dataset(**kwargs)
        elif 'UAV20' in name:
            dataset = UAV20Dataset(**kwargs)
        elif 'DTB70' in name:
            dataset = DTBDataset(**kwargs)
        elif 'UAVDT' in name:
            dataset = UAVDTDataset(**kwargs)
        elif 'VISDRONED' in name:
            dataset = VISDRONEDDataset(**kwargs)
        elif 'UAVTrack112' in name:
            dataset = V4RDataset(**kwargs)
        elif 'UAVDark' in name:
            dataset = UAVDARKDataset(**kwargs)
        elif 'DarkTrack2021' in name:
            dataset = DARKTRACKDataset(**kwargs)
        elif 'NAT2021' in name:
            dataset = DARKTRACKDataset(**kwargs)
        
        else:
            raise Exception("unknow dataset {}".format(kwargs['name']))
        return dataset
