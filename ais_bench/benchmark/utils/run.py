# flake8: noqa
# yapf: disable
from ctypes import ArgumentError
import os
from typing import List, Tuple, Union

import tabulate
from mmengine.config import Config

from ais_bench.benchmark.datasets.custom import make_custom_dataset_config
from ais_bench.benchmark.partitioners import NaivePartitioner, NumWorkerPartitioner, PerformancePartitioner
from ais_bench.benchmark.runners import LocalAPIRunner, LocalRunner
from ais_bench.benchmark.tasks import OpenICLEvalTask, OpenICLInferTask, OpenICLPerfTask, OpenICLInferMergedTask, OpenICLEvalMergedTask
from ais_bench.benchmark.openicl.icl_inferencer import (GenPerfInferencer, GenInferencer, GenMergedInferencer,
        GenPressureInferencer, GenModelPerfInferencer)
from ais_bench.benchmark.utils import get_logger, match_cfg_file

logger = get_logger()

def try_fill_in_custom_cfgs(config):
    for i, dataset in enumerate(config['datasets']):
        if 'type' not in dataset:
            config['datasets'][i] = make_custom_dataset_config(dataset)
    if 'model_dataset_combinations' not in config:
        return config
    for mdc in config['model_dataset_combinations']:
        for i, dataset in enumerate(mdc['datasets']):
            if 'type' not in dataset:
                mdc['datasets'][i] = make_custom_dataset_config(dataset)
    return config

def get_config_from_arg(args) -> Config:
    """Get the config object given args.

    Only a few argument combinations are accepted (priority from high to low)
    1. args.config
    2. args.models and args.datasets
    3. Huggingface parameter groups and args.datasets
    """

    if args.config:
        config = Config.fromfile(args.config, format_python_code=False)
        config = try_fill_in_custom_cfgs(config)
        return config

    # parse dataset args
    if not args.datasets and not args.custom_dataset_path:
        raise ValueError('You must specify "--datasets" or "--custom-dataset-path" if you do not specify a config file path.')
    datasets = []
    if args.datasets:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        datasets_dir = [
            os.path.join(args.config_dir, 'datasets'),
            os.path.join(args.config_dir, 'dataset_collections'),
            os.path.join(default_configs_dir, './datasets'),
            os.path.join(default_configs_dir, './dataset_collections')

        ]
        for dataset_arg in args.datasets:
            if '/' in dataset_arg:
                dataset_name, dataset_suffix = dataset_arg.split('/', 1)
                dataset_key_suffix = dataset_suffix
            else:
                dataset_name = dataset_arg
                dataset_key_suffix = '_datasets'

            for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                logger.info(f'Loading {dataset[0]}: {dataset[1]}')
                cfg = Config.fromfile(dataset[1])
                for k in cfg.keys():
                    if k.endswith(dataset_key_suffix):
                        datasets += cfg[k]
    else:
        dataset = {'path': args.custom_dataset_path}
        if args.custom_dataset_infer_method is not None:
            dataset['infer_method'] = args.custom_dataset_infer_method
        if args.custom_dataset_data_type is not None:
            dataset['data_type'] = args.custom_dataset_data_type
        if args.custom_dataset_meta_path is not None:
            dataset['meta_path'] = args.custom_dataset_meta_path
        dataset = make_custom_dataset_config(dataset)
        datasets.append(dataset)

    # parse model args
    if not args.models:
        raise ValueError('You must specify a config file path, or specify --models and --datasets.')
    models = []
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    default_configs_dir = os.path.join(parent_dir, 'configs')
    models_dir = [
        os.path.join(args.config_dir, 'models'),
        os.path.join(default_configs_dir, './models'),

    ]
    if args.models:
        for model_arg in args.models:
            for model in match_cfg_file(models_dir, [model_arg]):
                logger.info(f'Loading {model[0]}: {model[1]}')
                cfg = Config.fromfile(model[1])
                if 'models' not in cfg:
                    raise ValueError(f'Config file {model[1]} does not contain "models" field')
                models += cfg['models']
    else:
        raise ValueError('You must specify "--models"')

    # parse summarizer args
    summarizer_arg = args.summarizer if args.summarizer is not None else 'example'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    default_configs_dir = os.path.join(parent_dir, 'configs')
    summarizers_dir = [
        os.path.join(args.config_dir, 'summarizers'),
        os.path.join(default_configs_dir, './summarizers'),

    ]

    # Check if summarizer_arg contains '/'
    if '/' in summarizer_arg:
        # If it contains '/', split the string by '/'
        # and use the second part as the configuration key
        summarizer_file, summarizer_key = summarizer_arg.split('/', 1)
    else:
        # If it does not contain '/', keep the original logic unchanged
        summarizer_key = 'summarizer'
        summarizer_file = summarizer_arg

    s = match_cfg_file(summarizers_dir, [summarizer_file])[0]
    logger.info(f'Loading {s[0]}: {s[1]}')
    cfg = Config.fromfile(s[1])
    # Use summarizer_key to retrieve the summarizer definition
    # from the configuration file
    summarizer = cfg[summarizer_key]

    return Config(dict(models=models, datasets=datasets, summarizer=summarizer), format_python_code=False)


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



def fill_perf_cfg(cfg, args):
    models_attr = get_models_attr(cfg)
    if models_attr == "service":
        if args.disable_cb:
            logger.warning("disable_cb is not supported in perf mode, it will be ignored.")
        new_cfg = dict(infer=dict(
            partitioner=dict(type=get_config_type(PerformancePartitioner)),
            runner=dict(
                max_num_workers=args.max_num_workers,
                num_prompts=args.num_prompts,
                debug=args.debug,
                disable_cb=False,
                task=dict(type=get_config_type(OpenICLPerfTask)),
                type=get_config_type(LocalAPIRunner)
            )), )
        for data_config in cfg['datasets']:
            if args.pressure:
                data_config['infer_cfg']['inferencer']['type'] = get_config_type(GenPressureInferencer)
            else:
                data_config['infer_cfg']['inferencer']['type'] = get_config_type(GenPerfInferencer)
    else:
        new_cfg = dict(infer=dict(
            partitioner=dict(type=get_config_type(PerformancePartitioner)),
            runner=dict(
                max_num_workers=args.max_num_workers,
                max_workers_per_gpu=args.max_workers_per_gpu,
                debug=args.debug,
                task=dict(type=get_config_type(OpenICLPerfTask)),
                type=get_config_type(LocalRunner)
            )), )
        for data_config in cfg['datasets']:
            data_config['infer_cfg']['inferencer']['disable_cb'] = True
            data_config['infer_cfg']['inferencer']['type'] = get_config_type(GenModelPerfInferencer)

    cfg.merge_from_dict(new_cfg)


def fill_merged_infer_cfg(cfg, args):
    models_attr = get_models_attr(cfg)
    if models_attr == "service":
        new_cfg = dict(infer=dict(
            partitioner=dict(type=get_config_type(PerformancePartitioner)),
            runner=dict(
                max_num_workers=args.max_num_workers,
                debug=args.debug,
                disable_cb=args.disable_cb,
                task=dict(type=get_config_type(OpenICLInferMergedTask)),
                type=get_config_type(LocalAPIRunner),
            )), )
        if args.disable_cb:
            for data_config in cfg['datasets']:
                data_config['infer_cfg']['inferencer']['disable_cb'] = True
    else:
        new_cfg = dict(infer=dict(
            partitioner=dict(type=get_config_type(PerformancePartitioner)),
            runner=dict(
                max_num_workers=args.max_num_workers,
                max_workers_per_gpu=args.max_workers_per_gpu,
                debug=args.debug,
                task=dict(type=get_config_type(OpenICLInferMergedTask)),
                type=get_config_type(LocalRunner)
            )), )
        for data_config in cfg['datasets']:
            data_config['infer_cfg']['inferencer']['disable_cb'] = True

    for data_config in cfg['datasets']:
        data_config['infer_cfg']['inferencer']['type'] = get_config_type(GenMergedInferencer)

    cfg.merge_from_dict(new_cfg)


def fill_infer_cfg(cfg, args):
    models_attr = get_models_attr(cfg)
    if models_attr == "service":
        new_cfg = dict(infer=dict(
            partitioner=dict(type=get_config_type(NaivePartitioner)),
            runner=dict(
                max_num_workers=args.max_num_workers,
                debug=args.debug,
                disable_cb=args.disable_cb,
                task=dict(type=get_config_type(OpenICLInferTask)),
                type=get_config_type(LocalAPIRunner),
            )), )
        if args.disable_cb:
            for data_config in cfg['datasets']:
                data_config['infer_cfg']['inferencer']['disable_cb'] = True
    else:
        new_cfg = dict(infer=dict(
            partitioner=dict(type=get_config_type(NaivePartitioner)),
            runner=dict(
                max_num_workers=args.max_num_workers,
                max_workers_per_gpu=args.max_workers_per_gpu,
                debug=args.debug,
                task=dict(type=get_config_type(OpenICLInferTask)),
                type=get_config_type(LocalRunner),
            )), )
        for data_config in cfg['datasets']:
            data_config['infer_cfg']['inferencer']['disable_cb'] = True

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

def fill_merged_eval_cfg(cfg, args):
    new_cfg = dict(eval=dict(
        partitioner=dict(type=get_config_type(PerformancePartitioner)),
        runner=dict(
            max_num_workers=args.max_num_workers,
            debug=args.debug,
            task=dict(type=get_config_type(OpenICLEvalMergedTask)),
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
