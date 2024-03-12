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
from ais_bench.evaluate.interface import dataset_factory
from ais_bench.evaluate.dataset.base_dataset import BaseDataset
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
    def is_initialization_success(dataset_name):
        dataset_instance = dataset_factory.get(dataset_name)
        assert isinstance(dataset_instance, BaseDataset)

    def init(self):
        self.dataset_names = [CEVAL, MMLU, GSM8K]

    def test_success_initialization(self):
        for dataset_name in self.dataset_names:
            self.is_initialization_success(dataset_name)

    def test_failed_initialization_incorrect_name(self):
        dataset_name = "xxx"
        dataset_instance = dataset_factory.get(dataset_name)
        assert dataset_instance is None

    def test_failed_initialization_incorrect_shot(self):
        dataset_name = CEVAL
        dataset_instance = dataset_factory.get(dataset_name, shot=-1)
        assert dataset_instance is None

    def test_iterable(self):
        dataset_name = CEVAL
        dataset_instance = dataset_factory.get(dataset_name)

        for entry in dataset_instance:
            assert isinstance(entry, dict)
            assert INPUT_INDEX in entry
            assert TARGET_OUTPUT_INDEX in entry
            break

