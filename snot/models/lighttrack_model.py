# Parts of this code come from https://github.com/researchmm/LightTrack
from snot.models.lighttrack.backbone import build_subnet
from snot.models.lighttrack.backbone import build_supernet_DP
from snot.models.lighttrack.super_connect import head_supernet, MC_BN, Point_Neck_Mobile_simple_DP
from snot.models.lighttrack.super_model_DP import Super_model_DP, Super_model_DP_retrain
from snot.models.lighttrack.submodels import build_subnet_head, build_subnet_BN, build_subnet_feat_fusor
from snot.utils.utils_lighttrack import name2path



class LightTrackM_Supernet(Super_model_DP):
    def __init__(self, search_size=256, template_size=128, stride=16, adj_channel=128, build_module=True):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Supernet, self).__init__(search_size=search_size, template_size=template_size,
                                                   stride=stride)  # ATTENTION
        # config #
        # which parts to search
        self.search_back, self.search_ops, self.search_head = 1, 1, 1
        # backbone config
        self.stage_idx = [1, 2, 3]  # which stages to use
        self.max_flops_back = 470
        # head config
        self.channel_head = [128, 192, 256]
        self.kernel_head = [3, 5, 0]  # 0 means skip connection
        self.tower_num = 8  # max num of layers in the head
        self.num_choice_channel_head = len(self.channel_head)
        self.num_choice_kernel_head = len(self.kernel_head)
        # Compute some values #
        self.in_c = [self.channel_back[idx] for idx in self.stage_idx]
        strides_use = [self.strides[idx] for idx in self.stage_idx]
        strides_use_new = []
        for item in strides_use:
            if item not in strides_use_new:
                strides_use_new.append(item)  # remove repeated elements
        self.strides_use_new = strides_use_new
        self.num_kernel_corr = [int(round(template_size / stride) ** 2) for stride in strides_use_new]
        # build the architecture #
        if build_module:
            self.features, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)
            self.neck = MC_BN(inp_c=self.in_c)  # BN with multiple types of input channels
            self.feature_fusor = Point_Neck_Mobile_simple_DP(num_kernel_list=self.num_kernel_corr, matrix=True,
                                                             adj_channel=adj_channel)  # stride=8, stride=16
            self.supernet_head = head_supernet(channel_list=self.channel_head, kernel_list=self.kernel_head,
                                               linear_reg=True, inchannels=adj_channel, towernum=self.tower_num)
        else:
            _, self.sta_num = build_supernet_DP(flops_maximum=self.max_flops_back)


class LightTrackM_Subnet(Super_model_DP_retrain):
    def __init__(self, path_name, search_size=256, template_size=128, stride=16, adj_channel=128):
        """subclass calls father class's __init__ func"""
        super(LightTrackM_Subnet, self).__init__(search_size=search_size, template_size=template_size,
                                                 stride=stride)  # ATTENTION
        model_cfg = LightTrackM_Supernet(search_size=search_size, template_size=template_size,
                                         stride=stride, adj_channel=adj_channel, build_module=False)

        path_backbone, path_head, path_ops = name2path(path_name, sta_num=model_cfg.sta_num)
        # build the backbone
        self.features = build_subnet(path_backbone, ops=path_ops)  # sta_num is based on previous flops
        # build the neck layer
        self.neck = build_subnet_BN(path_ops, model_cfg)
        # build the Correlation layer and channel adjustment layer
        self.feature_fusor = build_subnet_feat_fusor(path_ops, model_cfg, matrix=True, adj_channel=adj_channel)
        # build the head
        self.head = build_subnet_head(path_head, channel_list=model_cfg.channel_head, kernel_list=model_cfg.kernel_head,
                                      inchannels=adj_channel, linear_reg=True, towernum=model_cfg.tower_num)
