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

from ais_bench.evaluate.interface import Filter
import os

target_output = ['The quick brown dog jumps on the log.', "This is a good translation."]
model_output = ['The fast black dog jumps over the log', "Can not understand this translation."]
indexs = [1, 2]

result_dict = {"indexs": indexs, "target_output": target_output, "model_output": model_output}
out_dir = os.path.dirname(__file__)

filter = Filter()

filter.add_measurement("bleu") # use default extra parameter ngram=1
filter.add_measurement("distinct", ngram=3)

filter.do_measuring(result_dict)

filter.do_filtering(out_dir) # use default threshold
filter.do_filtering(out_dir, thresholds={"bleu-1": 0.7, "distinct-3": 0.7})