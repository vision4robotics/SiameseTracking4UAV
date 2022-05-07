# -*- coding: utf-8 -*
from collections import OrderedDict
from typing import Dict, List

from yacs.config import CfgNode

from snot.models.fcpp.model.loss.loss_base import TASK_LOSSES
from snot.utils.utils_fcpp import merge_cfg_into_hps


def build(task: str, cfg: CfgNode):
    MODULES = TASK_LOSSES[task]

    names = cfg.names
    loss_dict = OrderedDict()
    for name in names:
        assert name in MODULES, "loss {} not registered for {}!".format(
            name, task)
        module = MODULES[name]()
        hps = module.get_hps()
        hps = merge_cfg_into_hps(cfg[name], hps)
        module.set_hps(hps)
        module.update_params()
        loss_dict[cfg[name].name] = module

    return loss_dict


def get_config(task_list: List) -> Dict[str, CfgNode]:
    cfg_dict = {name: CfgNode() for name in task_list}

    for cfg_name, modules in TASK_LOSSES.items():
        cfg = cfg_dict[cfg_name]
        cfg["names"] = list()
        for name in modules:
            cfg[name] = CfgNode()
            backbone = modules[name]
            hps = backbone.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
