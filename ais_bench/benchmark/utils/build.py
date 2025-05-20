import copy

from mmengine.config import ConfigDict

from ais_bench.benchmark.registry import LOAD_DATASET, MODELS, PERF_METRIC_CALCULATORS


def build_dataset_from_cfg(dataset_cfg: ConfigDict):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.pop('infer_cfg', None)
    dataset_cfg.pop('eval_cfg', None)
    dataset_cfg.pop('abbr', None)
    return LOAD_DATASET.build(dataset_cfg)


def build_model_from_cfg(model_cfg: ConfigDict):
    model_cfg = copy.deepcopy(model_cfg)
    model_cfg.pop('run_cfg', None)
    model_cfg.pop('max_out_len', None)
    model_cfg.pop('batch_size', None)
    model_cfg.pop('abbr', None)
    model_cfg.pop('attr', None)
    model_cfg.pop('summarizer_abbr', None)
    model_cfg.pop('pred_postprocessor', None)
    model_cfg.pop('min_out_len', None)
    return MODELS.build(model_cfg)


def build_perf_metric_calculator_from_cfg(metric_cfg: ConfigDict, perf_details: dict):
    metric_cfg = copy.deepcopy(metric_cfg)
    metric_cfg['perf_details'] = perf_details
    return PERF_METRIC_CALCULATORS.build(metric_cfg)