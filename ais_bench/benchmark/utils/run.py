# flake8: noqa
# yapf: disable
from ctypes import ArgumentError
import os
from typing import List, Tuple, Union

import tabulate
from mmengine.config import Config

from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners import LocalRunner
from ais_bench.benchmark.tasks import OpenICLEvalTask, OpenICLInferTask, OpenICLApiInferTask
from ais_bench.benchmark.openicl.icl_inferencer import  GenInferencer, GenModelPerfInferencer
from ais_bench.benchmark.utils import get_logger, match_cfg_file
from ais_bench.benchmark.datasets.custom import make_custom_dataset_config

logger = get_logger()

def try_fill_in_custom_cfgs(config):
    return config


def get_config_type(obj) -> str:
    return f'{obj.__module__}.{obj.__name__}'


def get_models_attr(cfg):
    attr_list = []
    for model_cfg in cfg['models']:
        attr = model_cfg.get('attr', 'service') # default service
        if attr not in ['local', 'service']:
            raise ValueError(f"Model config contain illegal attr, model abbr is {model_cfg.get('abbr')}")
        if attr not in attr_list:
            attr_list.append(attr)

    if len(attr_list) != 1:
        raise ValueError("Cannot run local and service model together! Please check parameters of --models!")

    return attr_list[0]


def fill_infer_cfg(cfg, args):
    new_cfg = dict(infer=dict(
        partitioner=dict(type=get_config_type(NaivePartitioner)),
        runner=dict(
            max_num_workers=args.max_num_workers,
            max_workers_per_gpu=args.max_workers_per_gpu,
            debug=args.debug,
            task=dict(type=get_config_type(OpenICLApiInferTask)),
            type=get_config_type(LocalRunner),
        )), )
    for data_config in cfg['datasets']:
        retriever_cfg = data_config['infer_cfg']['retriever']
        infer_cfg = data_config['infer_cfg']
        if "prompt_template" in infer_cfg:
            retriever_cfg["prompt_template"] = infer_cfg["prompt_template"]
        if "ice_template" in infer_cfg:
            retriever_cfg["ice_template"] = infer_cfg["ice_template"]

    cfg.merge_from_dict(new_cfg)


def fill_eval_cfg(cfg, args):
    new_cfg = dict(eval=dict(
        partitioner=dict(type=get_config_type(NaivePartitioner)),
        runner=dict(
            max_num_workers=args.max_num_workers,
            debug=args.debug,
            task=dict(type=get_config_type(OpenICLEvalTask)),
        )), )

    new_cfg['eval']['runner']['type'] = get_config_type(LocalRunner)
    new_cfg['eval']['runner']['max_workers_per_gpu'] = args.max_workers_per_gpu
    cfg.merge_from_dict(new_cfg)


def function_call_task_check(cfg, merge_ds):
    """
    Check if the configuration represents a function call task.

    A function call task is defined as having all models of type VLLMFunctionCallAPIChat
    and all datasets of type BFCLDataset. If there's any mixing of incompatible types
    (e.g., BFCLDataset with non-VLLMFunctionCallAPIChat models), a ValueError is raised.

    Args:
        cfg: Configuration object containing 'models' and 'datasets' lists

    Raises:
        ValueError: If there's an incompatible combination of model and dataset types

    Returns:
        None: Modifies the cfg object by adding 'is_function_call_task' boolean flag
    """
    vllm_function_call_type = 'ais_bench.benchmark.models.VLLMFunctionCallAPIChat'
    bfcl_dataset_type = 'ais_bench.benchmark.datasets.BFCLDataset'

    all_models_function_call = True
    for model_cfg in cfg['models']:
        if model_cfg.get('type') != vllm_function_call_type:
            all_models_function_call = False
            break

    all_datasets_bfcl = True
    for data_cfg in cfg['datasets']:
        if data_cfg.get('type') != bfcl_dataset_type:
            all_datasets_bfcl = False
            break

    has_bfcl_dataset = any(data_cfg.get('type') == bfcl_dataset_type for data_cfg in cfg['datasets'])
    has_function_call_model = any(model_cfg.get('type') == vllm_function_call_type for model_cfg in cfg['models'])

    if has_bfcl_dataset and not all_models_function_call:
        non_function_call_models = [model_cfg.get('type').split('.')[-1] for model_cfg in cfg['models']
                                  if model_cfg.get('type') != vllm_function_call_type]
        raise ValueError(f"BFCLDataset can only be used with VLLMFunctionCallAPIChat, but found incompatible models: {non_function_call_models}")

    if has_function_call_model and not all_datasets_bfcl:
        non_bfcl_datasets = [data_cfg.get('type').split('.')[-1] for data_cfg in cfg['datasets']
                            if data_cfg.get('type') != bfcl_dataset_type]
        raise ValueError(f"VLLMFunctionCallAPIChat can only be used with BFCLDataset, but found incompatible datasets: {non_bfcl_datasets}")

    is_function_call_task = all_models_function_call and all_datasets_bfcl
    if is_function_call_task and merge_ds:
        raise ValueError("Option '--merge-ds' is not supported with function call tasks")
    cfg.merge_from_dict({"is_function_call_task": is_function_call_task})
