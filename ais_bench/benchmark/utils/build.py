import re
import os
import copy
import ipaddress

from mmengine.config import ConfigDict

from ais_bench.benchmark.registry import LOAD_DATASET, MODELS, CLIENTS, PERF_METRIC_CALCULATORS


def validate_model_cfg(model_cfg: dict) -> dict:
    errors = {}

    def check(condition, key, message):
        if not condition:
            errors[key] = message

    validators = {
        "attr": lambda v: (v in ("local", "service"), "attr must be 'local' or 'service'"),
        "abbr": lambda v: (isinstance(v, str) and re.fullmatch(r'[A-Za-z\-]+', v),
                           "abbr must contain only letters and hyphens (e.g., 'vllm-api-general-chat')"),
        "path": lambda v: (not v or (isinstance(v, str) and os.path.exists(v)),
                           f"path is not accessible or does not exist: {v}"),
        "model": lambda v: (not v or isinstance(v, str),
                            "model can be omitted or must be a string"),
        "request_rate": lambda v: (isinstance(v, (int, float)) and 0 <= v <= 64000,
                                   "request_rate must be a number in the range [0, 64000]"),
        "retry": lambda v: (isinstance(v, int) and 0 <= v <= 1000,
                            "retry must be an integer in the range [0, 1000]"),
        "host_port": lambda v: (isinstance(v, int) and (0 < v < 65536),
                                "host_port must be a valid port number in the range (0, 65536)"),
        "max_out_len": lambda v: (isinstance(v, int) and 0 < v <= 131072,
                                  "max_out_len must be an integer in the range (0, 131072]"),
        "batch_size": lambda v: (isinstance(v, int) and 0 < v <= 64000,
                                 "batch_size must be an integer in the range (0, 64000]"),
        "generation_kwargs": lambda v: (isinstance(v, dict),
                                        "generation_kwargs must be a dictionary"),
    }

    for key, value in model_cfg.items():
        if key == "type":
            continue  # Not configurable; skip validation

        if key == "host_ip":
            if value == "localhost":
                continue
            try:
                ipaddress.ip_address(value)
            except ValueError:
                errors[key] = "host_ip must be a valid IPv4 or IPv6 address"
            continue

        validator = validators.get(key)
        if validator:
            valid, msg = validator(value)
            check(valid, key, msg)

    return errors
    

def build_dataset_from_cfg(dataset_cfg: ConfigDict):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.pop('infer_cfg', None)
    dataset_cfg.pop('eval_cfg', None)
    dataset_cfg.pop('abbr', None)
    return LOAD_DATASET.build(dataset_cfg)


def build_synthetic_dataset_from_cfg(dataset_cfg: ConfigDict, model_cfg: ConfigDict):
    dataset_cfg = copy.deepcopy(dataset_cfg)
    dataset_cfg.pop('infer_cfg', None)
    dataset_cfg.pop('eval_cfg', None)
    dataset_cfg.pop('abbr', None)
    dataset_cfg.update({'model_path' : model_cfg.get("path", "None")})
    return LOAD_DATASET.build(dataset_cfg)


def build_model_from_cfg(model_cfg: ConfigDict):
    model_cfg = copy.deepcopy(model_cfg)
    model_name = model_cfg.get("type", "").split(".")[-1]
    errors = validate_model_cfg(model_cfg)
    if errors:
        raise ValueError(f"{model_name} build failed with the following errors: {errors}")
    model_cfg.pop('run_cfg', None)
    model_cfg.pop('max_out_len', None)
    model_cfg.pop('batch_size', None)
    model_cfg.pop('abbr', None)
    model_cfg.pop('attr', None)
    model_cfg.pop('summarizer_abbr', None)
    model_cfg.pop('pred_postprocessor', None)
    model_cfg.pop('min_out_len', None)
    return MODELS.build(model_cfg)

def build_client_from_cfg(client_cfg: ConfigDict):
    return CLIENTS.build(client_cfg)

def build_perf_metric_calculator_from_cfg(metric_cfg: ConfigDict):
    metric_cfg = copy.deepcopy(metric_cfg)
    return PERF_METRIC_CALCULATORS.build(metric_cfg)
