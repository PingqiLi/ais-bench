# Copyright (c) 2025-2025 Huawei Technologies Co., Ltd.
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
import pytest
import shutil
from ais_bench.benchmark.utils.datasets import get_data_path

GSK8K_DATA_COUNT = 1319


class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):

        print('\n ---class level teardown_class')

    def init(self):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_path = os.path.abspath(os.path.join(self.cur_dir, "../testdatas"))
        if os.path.exists(self.test_data_path):
            shutil.rmtree(self.test_data_path)
        os.makedirs(self.test_data_path)
        self.datasets_dir = os.path.join(self.test_data_path, "datasets")
        os.makedirs(self.datasets_dir)


    # mode infer
    def test_customized_dataset_is_not_an_abspath(self, monkeypatch):
        dataset_name = "gsm8k"
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        if os.path.exists(dataset_path):
            shutil.rmtree(dataset_path)
        monkeypatch.setattr("ais_bench.benchmark.utils.datasets.get_cache_dir", lambda *arg: self.datasets_dir)

        with pytest.raises(TypeError) as e:
            get_data_path(dataset_name, local_mode=False)
            assert "Customized dataset path type is not a absolute path" in e

