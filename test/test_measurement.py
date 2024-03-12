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

import sys
import logging

import pytest
from ais_bench.evaluate.interface import measurement_factory
from ais_bench.evaluate.measurement.measurement import *
from ais_bench.evaluate.common.global_var import *

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @staticmethod
    def is_initialization_success(measurement_name):
        measurement_instance = measurement_factory.get(measurement_name)
        assert isinstance(measurement_instance, BaseMeasurement)

    @staticmethod
    def check_measurement(measurement_name, target_output, model_output, target_metrics_list):
        measurement_instance = measurement_factory.get(measurement_name)
        assert isinstance(measurement_instance, BaseMeasurement)

        indexs = list(range(len(target_output)))
        result_dict = {INDEX_INDEX: indexs, TARGET_OUTPUT_INDEX: target_output, MODEL_OUTPUT_INDEX: model_output}

        model_metrics = measurement_instance(result_dict)
        metrics_list = measurement_instance.get_metrics_list()
        for metrics, target_metrics in zip(metrics_list, target_metrics_list):
            assert abs(metrics - target_metrics) < EPSILON
        assert abs(model_metrics - sum(target_metrics_list) / len(metrics_list)) < EPSILON

    def init(self):
        self.measurement_names = [ACCURACY, EDIT_DISTANCE, BLEU, ROUGE, ABNORMAL_STRING_RATE,
                                  DISTINCT, RELATIVE_DISTINCT, RELATIVE_ABNORMAL_STRING_RATE]

    def test_initialization_success(self):
        for measurement_name in self.measurement_names:
            self.is_initialization_success(measurement_name)

    def test_initialization_failed_incorrect_name(self):
        measurement_name = "xxx"
        measurement_instance = measurement_factory.get(measurement_name)
        assert measurement_instance is None

    def test_accuracy(self):
        measurement_name = ACCURACY
        target_output = ["A", "B", "C", "D"]
        model_output = ["A", "B", "B", "C"]
        target_metrics_list = [1, 1, 0, 0]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)

    def test_edit_distance(self):
        measurement_name = EDIT_DISTANCE
        target_output = ["dog", "dog"]
        model_output = ["god", "dogs"]
        target_metrics_list = [2, 1]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)

    def test_bleu(self):
        measurement_name = BLEU
        target_output = ['The quick brown dog jumps on the log.', "This is a good translation."]
        model_output = ['The fast black dog jumps over the log', "Can not understand this translation."]
        target_metrics_list = [0.625, 0.2]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)

    def test_rouge(self):
        measurement_name = ROUGE
        target_output = ['The fast brown dog runs', "This is a good translation."]
        model_output = ['The fast black dog runs', "Can not understand this translation."]
        target_metrics_list = [0.8, 1/3]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)

    def test_distinct(self):
        measurement_name = DISTINCT
        target_output = ['The quick brown dog jumps on the log the log', "This is a good translation."]
        model_output = ['The black dog jumps over the log the log', "Can not understand this translation."]
        target_metrics_list = [0.875, 1.0]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)

    def test_relative_distinct(self):
        measurement_name = RELATIVE_DISTINCT
        target_output = ['The quick brown dog jumps on the log the log', "This is a good translation."]
        model_output = ['The black dog jumps over the log the log', "Can not understand this translation."]
        target_metrics_list = [0.984375, 1.0]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)

    def test_abnormal_string_rate(self):
        measurement_name = ABNORMAL_STRING_RATE
        target_output = ['The quick brown dog jumps on the log the log', "This is a good translation."]
        model_output = ['The quick brown dog jumps on the log $ $', "Can not understand this translation."]
        target_metrics_list = [0.2, 0.0]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)

    def test_relative_abnormal_string_rate(self):
        measurement_name = RELATIVE_ABNORMAL_STRING_RATE
        target_output = ['The quick brown dog jumps on $ $ $ $', "This is a good translation."]
        model_output = ['The quick brown dog jumps on the log $ $', "Can not understand this translation."]
        target_metrics_list = [0.5, 0.0]

        self.check_measurement(measurement_name, target_output, model_output, target_metrics_list)