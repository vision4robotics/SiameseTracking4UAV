from typing import Union
from yacs.config import CfgNode

from snot.models.fcpp.engine.tester.builder import get_config as get_tester_cfg
from snot.models.fcpp.model.builder import get_config as get_model_cfg
from snot.models.fcpp.pipeline.builder import get_config as get_pipeline_cfg

cfg = CfgNode()  # root_cfg
task_list = ["track",]
default_str = "unknown"
cfg["task_name"] = default_str

# default configuration for test
cfg["test"] = CfgNode()
test_cfg = cfg["test"]
for task in task_list:
    test_cfg[task] = CfgNode()
    test_cfg[task]["exp_name"] = default_str
    test_cfg[task]["exp_save"] = default_str
    test_cfg[task]["model"] = get_model_cfg(task_list)[task]
    test_cfg[task]["pipeline"] = get_pipeline_cfg(task_list)[task]
    test_cfg[task]["tester"] = get_tester_cfg(task_list)[task]



def specify_task(cfg: CfgNode) -> Union[str, CfgNode]:
    r"""
    get task's short name from config, and specify task config

    Args:
        cfg (CfgNode): config
        
    Returns:
        short task name, task-specified cfg
    """
    for task in task_list:
        if cfg[task]['exp_name'] != default_str:
            return task, cfg[task]
    assert False, "unknown task!"
