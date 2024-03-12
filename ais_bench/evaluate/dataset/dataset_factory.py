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

from ais_bench.evaluate.dataset.ceval_dataset import CevalDataset
from ais_bench.evaluate.dataset.mmlu_dataset import MmluDataset
from ais_bench.evaluate.dataset.gsm8k_dataset import Gsm8kDataset
from ais_bench.evaluate.common.log import logger
from ais_bench.evaluate.common.global_var import *

dataset_switch = {
    CEVAL: CevalDataset,
    MMLU: MmluDataset,
    GSM8K: Gsm8kDataset
}


class DatasetFactory():
    @staticmethod
    def get(dataset_name, dataset_path=None, shot=0):
        is_dataset_name_formated = dataset_switch.get(dataset_name.strip()) is not None
        is_dataset_path_formated = (dataset_path is None or isinstance(dataset_path, str))
        is_shot_formated = (isinstance(shot, int) and shot >= 0 and shot <= 5)
        if is_dataset_name_formated and is_dataset_path_formated and is_shot_formated:
            try:
                dataset_instance = dataset_switch.get(dataset_name.strip())(dataset_name, dataset_path, shot)
            except ValueError as e:
                logger.error(f"get dataset instance failed, exact exception: {e}")
                dataset_instance = None
        else:
            logger.error(f"Please check the input. "
                         f"Currently only {', '.join(list(dataset_switch.keys()))} dataset are supported. "
                         f"If shot or dataset_path is set, check them: "
                         f"shot should be of type int and between 0 and 5, "
                         f"dataset_path should be string.")
            dataset_instance = None
        return dataset_instance

dataset_factory = DatasetFactory()