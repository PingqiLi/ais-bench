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
import os

import pytest
from ais_bench.evaluate.interface import Filter
from ais_bench.evaluate.common.global_var import *

logging.basicConfig(stream = sys.stdout, level = logging.INFO, format = '[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


class TestClass:
    def test_filter(self):
        target_output = ['The quick brown dog jumps on the log.', "This is a good translation."]
        model_output = ['The fast black dog jumps over the log', "Can not understand this translation."]
        indexs = [1, 2]

        result_dict = {INDEX_INDEX: indexs, TARGET_OUTPUT_INDEX: target_output, MODEL_OUTPUT_INDEX: model_output}
        out_dir = os.path.dirname(__file__)

        filter = Filter()

        filter.add_measurement(BLEU) # use default extra parameter ngram=1
        filter.add_measurement(DISTINCT, ngram=3)

        filter.do_measuring(result_dict)

        filter.do_filtering(out_dir) # use default threshold
        filter.do_filtering(out_dir, thresholds={BLEU: 0.7, DISTINCT: 0.7})