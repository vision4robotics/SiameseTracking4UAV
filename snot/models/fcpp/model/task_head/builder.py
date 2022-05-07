# -*- coding: utf-8 -*
from typing import Dict, List

from yacs.config import CfgNode

from snot.models.fcpp.model.task_head.taskhead_base import TASK_HEADS
from snot.utils.utils_fcpp import merge_cfg_into_hps


def build(task: str, cfg: CfgNode):
    r"""
    Builder function.

    Arguments
    ---------
    task: str
        builder task name (track|vos)
    cfg: CfgNode
        buidler configuration

    Returns
    -------
    torch.nn.Module
        module built by builder
    """

    head_modules = TASK_HEADS[task]

    name = cfg.name
    head_module = head_modules[name]()
    hps = head_module.get_hps()
    hps = merge_cfg_into_hps(cfg[name], hps)
    head_module.set_hps(hps)
    head_module.update_params()

    return head_module


def get_config(task_list: List) -> Dict[str, CfgNode]:
    r"""
    Get available component list config

    Returns
    -------
    Dict[str, CfgNode]
        config with list of available components
    """
    cfg_dict = {task: CfgNode() for task in task_list}
    for cfg_name, module in TASK_HEADS.items():
        cfg = cfg_dict[cfg_name]
        cfg["name"] = "unknown"
        for name in module:
            cfg[name] = CfgNode()
            task_model = module[name]
            hps = task_model.default_hyper_params
            for hp_name in hps:
                cfg[name][hp_name] = hps[hp_name]
    return cfg_dict
