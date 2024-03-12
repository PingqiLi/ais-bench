# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from tqdm import tqdm
import pandas as pd

from ais_bench.evaluate.dataset.dataset_factory import dataset_factory
from ais_bench.evaluate.dataset.base_dataset import BaseDataset
from ais_bench.evaluate.common.global_var import *
from ais_bench.evaluate.measurement.measurement_factory import measurement_factory
from ais_bench.evaluate.measurement.measurement import BaseMeasurement
from ais_bench.evaluate.common.log import logger
from ais_bench.evaluate.common.path_check import ms_open, MAX_SIZE_LIMITED_NORMAL_FILE
from ais_bench.evaluate.common.utils import generate_random_string

class Evaluator():
    def __init__(self, generate_func, dataset_instance, measurement_instance, rank=0):
        self.check_generate_func(generate_func)
        self.check_dataset_instance(dataset_instance)
        self.check_measurement_instance(measurement_instance)
        self.generate = generate_func
        self.dataset = dataset_instance
        self.measurement = measurement_instance
        self.rank = rank

    @staticmethod
    def check_generate_func(generate_func):
        if callable(generate_func):
            all_args = generate_func.__code__.co_argcount
            kwargs = len(generate_func.__defaults__) if generate_func.__defaults__ else 0
            if all_args - kwargs != 1:
                logger.error("check_generate_func failed: number of function's required argument not equals to 1.")
                raise ValueError
        else:
            logger.error("check_generate_func failed: please pass a callable object.")
            raise ValueError

    @staticmethod
    def check_dataset_instance(dataset_instance):
        if not isinstance(dataset_instance, BaseDataset):
            logger.error("check_dataset_instance failed: dataset_instance is not a BaseDataset.")
            raise ValueError

    @staticmethod
    def check_measurement_instance(measurement_instance):
        if not isinstance(measurement_instance, BaseMeasurement):
            logger.error("check_measurement_instance failed: measurement_instance is not a BaseMeasurement.")
            raise ValueError

    def evaluate(self):
        logger.info(f"Start to evaluate on {self.dataset.dataset_name} dataset "
                    f"with {self.measurement.name} metric.")
        result = {
            INPUT_INDEX: [], TARGET_OUTPUT_INDEX: [], MODEL_OUTPUT_INDEX: [],
            INDEX_INDEX: [], CATEGORY_INDEX: [], METRICS_INDEX: []
        }
        metrics = 0.0

        for i, entry_dict in enumerate(tqdm(self.dataset)): # entry_dict must contains keys: input, target_output
            if not isinstance(entry_dict, dict) or INPUT_INDEX not in entry_dict or \
                TARGET_OUTPUT_INDEX not in entry_dict:
                logger.debug(f"{i}th entry is {entry_dict}, "
                             f"which is not dict or does not contain keys 'input' or 'target_output'")
                continue
            model_output = self.generate(entry_dict.get(INPUT_INDEX))
            if self.rank == 0:
                result[INPUT_INDEX].append(entry_dict.get(INPUT_INDEX))
                result[TARGET_OUTPUT_INDEX].append(entry_dict.get(TARGET_OUTPUT_INDEX))
                result[MODEL_OUTPUT_INDEX].append(model_output)
                result[INDEX_INDEX].append(entry_dict.get(INDEX_INDEX, i))
                result[CATEGORY_INDEX].append(entry_dict.get(CATEGORY_INDEX, ""))

        if self.rank == 0:
            metrics = self.measurement(result)
            result[METRICS_INDEX] = self.measurement.get_metrics_list()
            logger.info(f"The result of using {self.measurement.name} measurement "
                        f"to evaluate on {self.dataset.dataset_name} is: {metrics}")

        return metrics, result


class Filter():
    def __init__(self) -> None:
        self.measurements = []
        self.result_dict = None

    def add_measurement(self, measurement: str, **kwargs):
        measurement_method = measurement_factory.get(measurement, **kwargs)
        if measurement_method is not None:
            self.measurements.append(measurement_method)
        else:
            logger.warning(f"add {measurement} measurement failed")

    def do_measuring(self, result_dict):
        self.result_dict = result_dict
        logger.info("Start measuring.")
        for m in tqdm(self.measurements):
            m(result_dict)

    def do_filtering(self, out_dir, thresholds=None):
        if not isinstance(thresholds, dict):
            thresholds = dict()

        negative_indexs_set = set()
        logger.info("Start filtering.")
        for m in tqdm(self.measurements):
            threshold = thresholds.get(m.name, None)
            negative_indexs = m.filtering_helper(threshold)
            negative_indexs_set.update(negative_indexs)
        sorted_negative_indexs = sorted(negative_indexs_set)

        filtered_dict = dict()
        filtered_dict[INDEX_INDEX] = [self.result_dict[INDEX_INDEX][i] for i in sorted_negative_indexs]
        filtered_dict[MODEL_OUTPUT_INDEX] = [self.result_dict[MODEL_OUTPUT_INDEX][i] for i in sorted_negative_indexs]
        filtered_dict[TARGET_OUTPUT_INDEX] = [self.result_dict[TARGET_OUTPUT_INDEX][i] for i in sorted_negative_indexs]

        for m in self.measurements:
            filtered_dict[m.name] = [m.get_metrics_list()[i] for i in sorted_negative_indexs]

        df = pd.DataFrame(filtered_dict)
        columns=[INDEX_INDEX, MODEL_OUTPUT_INDEX, TARGET_OUTPUT_INDEX] + [m.name for m in self.measurements]
        df = df [columns]

        file_name = os.path.join(out_dir, str(os.getpid()) + "_" + generate_random_string() + "_filtering_result.csv")
        with ms_open(file_name, mode="w") as file:
            df.to_csv(file, index=False)
        return df


