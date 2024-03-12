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
import json
import pandas as pd

from ais_bench.evaluate.dataset.base_dataset import BaseDataset
from ais_bench.evaluate.measurement.measurement_factory import MeasurementFactory
from ais_bench.evaluate.common.log import logger
from ais_bench.evaluate.common.global_var import *
from ais_bench.evaluate.common.path_check import ms_open, MAX_SIZE_LIMITED_NORMAL_FILE

SUBCATEGORY_INDEX = 0
CHINESE_INDEX = 1
CATEGORY_INDEX = 2
VAL_INDEX = 3
PROMPT_INDEX = 4


class CevalDataset(BaseDataset):
    def __init__(self, dataset_name, dataset_path=None, shot=0) -> None:
        super().__init__(dataset_name, dataset_path, shot)
        self.current_key = 0
        self.current_index = 0
        self.global_index = 0

    def __len__(self):
        count = 0
        for value in self.subject_mapping.values():
            count += len(value[VAL_INDEX])
        return count

    def __iter__(self):
        self.current_key = 0
        self.current_index = 0
        self.global_index = 0
        return self


    def __next__(self):
        if self.current_key >= len(self.subjects):
            raise StopIteration

        key = self.subjects[self.current_key]
        subcategory = self.subject_mapping.get(key)[SUBCATEGORY_INDEX]
        category = self.subject_mapping.get(key)[CATEGORY_INDEX]
        val_df  = self.subject_mapping.get(key)[VAL_INDEX]
        prompt = self.subject_mapping.get(key)[PROMPT_INDEX]

        if self.current_index >= len(val_df):
            self.current_key += 1
            self.current_index = 0
            return self.__next__()

        val_row = val_df.loc[self.current_index]

        prompt += f"问： {val_row['question']}\nA. {val_row['A']}\nB. {val_row['B']}\n" \
            f"C. {val_row['C']}\nD. {val_row['D']}\n答：\n"
        result = {
            INDEX_INDEX: self.global_index, CATEGORY_INDEX: category + '/' + subcategory,
            INPUT_INDEX: prompt, TARGET_OUTPUT_INDEX: val_row["answer"]
        }
        self.current_index += 1
        self.global_index += 1
        return result

    def load(self):
        '''
        dataset_path-
        use val dataset as validation use dev dataset as prompt
        '''
        if self.dataset_path is None:
            self._download()
        self._check()

        subject_mapping_path = os.path.join(self.dataset_path, "subject_mapping.json")
        with ms_open(subject_mapping_path, max_size=MAX_SIZE_LIMITED_NORMAL_FILE) as file:
            self.subject_mapping = json.load(file)

        for subject in self.subject_mapping:
            val_path = os.path.join(self.dataset_path, "val", subject + "_val.csv")
            prompt_path = os.path.join(self.dataset_path, "dev", subject + "_dev.csv")
            with ms_open(val_path, max_size=MAX_SIZE_LIMITED_NORMAL_FILE) as file:
                val_df = pd.read_csv(file, header=0)
            self.subject_mapping[subject].append(val_df)
            with ms_open(prompt_path, max_size=MAX_SIZE_LIMITED_NORMAL_FILE) as file:
                prompt_df = pd.read_csv(file, header=0)[:self.shot + 1]
            prompt = self._gen_prompt(prompt_df, self.subject_mapping[subject][CHINESE_INDEX])
            self.subject_mapping[subject].append(prompt)
        self.subjects = list(self.subject_mapping.keys())

    def _gen_prompt(self, prompt_df, category_name):
        question_template = "问： {question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n答： {answer}\n"

        prompt = f"以下展示了在{category_name}领域的选择题及其正确答案\n\n"
        for _, row in prompt_df.iterrows():
            prompt += question_template.format(question=row["question"], A=row["A"], B=row["B"],
                                               C=row["C"], D=row["D"], answer=row["answer"])
        prompt += "请回答以下选择题\n"
        return prompt