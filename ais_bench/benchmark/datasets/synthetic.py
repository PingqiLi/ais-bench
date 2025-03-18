from typing import Dict, List
import json
import os
from typing import Any, Tuple
from dataclasses import dataclass

import numpy as np

from datasets import Dataset
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils import get_data_path
from .base import BaseDataset


@dataclass
class NumberRange:
    lower: tuple[int, float] = None
    upper: tuple[int, float] = None
    lower_inclusive: bool = True
    upper_inclusive: bool = True


def _check_keys_equal(got_keys, true_keys, comment):
    for key in got_keys:
        if key not in true_keys:
            raise ValueError(f"{key} is not a valid key for {comment}.")
    if got_keys != true_keys:
        raise ValueError(f"Expect keys {true_keys} for {comment}, but got keys {set(got_keys)}.")

def check_type(name: str, value: Any, types: Tuple):
    if not isinstance(value, types):
        raise ValueError(f"Parameter {name} should have type {types}, but got {type(value)}.")

def check_range(name: str, value: Any, param: NumberRange):
    lower, upper = param.lower, param.upper
    lower_inclusive, upper_inclusive = param.lower_inclusive, param.upper_inclusive
    # 构造区间的字符串表示
    lower_bound = '[' if lower_inclusive else '('
    upper_bound = ']' if upper_inclusive else ')'
    lower_str = str(lower) if lower is not None else '-inf'
    upper_str = str(upper) if upper is not None else '+inf'

    # 构造完整区间表示字符串
    interval_str = f"{lower_bound}{lower_str}, {upper_str}{upper_bound}"

    # 检查下限
    if lower is not None:
        lt = (lower_inclusive and value < lower)
        le = (not lower_inclusive and value <= lower)
        if le or lt:
            raise ValueError(f"Parameter {name} is {value}, not within the required range {interval_str}")
    # 检查上限
    if upper is not None:
        gt = (upper_inclusive and value > upper)
        ge = (not upper_inclusive and value >= upper)
        if gt or ge:
            raise ValueError(f"Parameter {name} is {value}, not within the required range {interval_str}")


@LOAD_DATASET.register_module()
class SyntheticDataset(BaseDataset):

    @staticmethod
    def _check_config_json(synthetic_config: Dict):
        input_str = "Input"
        output_str = "Output"
        request_count_str = "RequestCount"
        _check_keys_equal(synthetic_config.keys(), {input_str, output_str, request_count_str}, "SyntheticConfig")

        request_count = synthetic_config.get(request_count_str)
        check_type(request_count_str, request_count, (int,))
        check_range(request_count_str, request_count, NumberRange(0, 2**20))

        for key in (input_str, output_str):
            conf = synthetic_config.get(key)
            _check_keys_equal(conf.keys(), {"Method", "Params"}, 'SyntheticConfig["{key}"]')
            method = conf.get("Method")
            params = conf.get("Params")
            uniform_str = "uniform"
            gaussian_str = "gaussian"
            zipf_str = "zipf"
            min_value_str = "MinValue"
            max_value_str = "MaxValue"
            mean_str = "Mean"
            var_str = "Var"
            alpha_str = "Alpha"

            if method == uniform_str:
                _check_keys_equal(params.keys(), {min_value_str, max_value_str}, uniform_str)
            elif method == gaussian_str:
                _check_keys_equal(params.keys(), {mean_str, var_str, min_value_str, max_value_str}, gaussian_str)
            elif method == zipf_str:
                _check_keys_equal(params.keys(), {alpha_str, min_value_str, max_value_str}, zipf_str)
            else:
                raise ValueError(f'Method should be one of {{{uniform_str, gaussian_str, zipf_str}}}, '
                                 f'but got {method}.')
            min_float32_value = -3.0e38
            max_float32_value = 3.0e38
            for param_name, param_value in params.items():
                desc_name = key + " " + param_name
                if param_name in (min_value_str, max_value_str):
                    check_type(desc_name, param_value, types=(int,))
                    if key == input_str:
                        check_range(desc_name, param_value, NumberRange(1, 2**20))    # 2**20 = 1M
                    elif key == output_str:
                        check_range(desc_name, param_value, NumberRange(1, 2**20))
                elif param_name == mean_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(min_float32_value, max_float32_value))
                elif param_name == var_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(0, max_float32_value))
                elif param_name == alpha_str:
                    check_type(desc_name, param_value, types=(int, float))
                    check_range(desc_name, param_value, NumberRange(1.0, 10.0, lower_inclusive=False))
            min_value = params.get(min_value_str)
            max_value = params.get(max_value_str)
            if min_value > max_value:
                raise ValueError(f'MinValue should less than MaxValue, '
                                 f'but got MinValue is {min_value}, and MaxValue is {max_value}.')
    
    @staticmethod
    def sample_one_value(method: str, params: dict) -> int:
        # Sample one value, the args have been checked before
        min_value = params["MinValue"]
        max_value = params["MaxValue"]
        if method == "uniform":
            value = np.random.uniform(min_value, max_value)
        elif method == "gaussian":
            mean = params["Mean"]
            stddev = np.sqrt(params["Var"])
            value = np.random.normal(mean, stddev)
            value = np.clip(value, min_value, max_value)
        elif method == "zipf":
            alpha = params["Alpha"]
            value = np.random.zipf(alpha)
            value = np.clip(value, min_value, max_value)
        else:
            raise ValueError(f"Unknown method: {method}, method should be one of {{uniform, gaussian, zipf}}.")
        return int(value)
    
    @staticmethod
    def read_line(self, line: List[int]) -> Dict:
        """Get a data dict according to line.

        Args:
            line (List[int]): Input line should be a list with 2 integral elements, it represents the number of input
                token and output token respectively.

        Returns:
            A data string which contains 2 parts: "data", "max_new_tokens".
        """
        if not hasattr(line, '__len__') or len(line) != 2:
            raise ValueError("Input line should be a list with 2 integral elements.")
        default_str = "A"
        num_input_token, num_expect_generated_tokens = line
        data = " ".join([default_str] * num_input_token)
        return data + str(num_expect_generated_tokens)
        
    def load(self, path, **kwargs):
        path = get_data_path(path)
        path = os.path.join(path, "synthetic_config.json")
        dataset = []
        try:
            with open(path, mode="r", encoding="utf-8") as file:
                config = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError("Failed to load JSON config from `SyntheticConfigPath` file.") from e
        self._check_config_json(config)
        request_count = config.get("RequestCount")
        input_method = config["Input"]["Method"]
        input_params = config["Input"]["Params"]
        output_method = config["Output"]["Method"]
        output_params = config["Output"]["Params"]
        num_input_output_tokens = [[self.sample_one_value(input_method, input_params),
                                        self.sample_one_value(output_method, output_params)]
                                    for _ in range(request_count)]

        for num_input_output_token in num_input_output_tokens:
            data = self.read_line(self, num_input_output_token)
            dataset.append({"question":data,"answer":"aaa"})
        return Dataset.from_list(dataset)