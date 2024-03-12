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

from abc import abstractmethod, ABCMeta
from typing import Any
import logging
import re
import warnings
from collections import Counter

import nltk
from nltk.translate import bleu_score
import jieba
from rouge_chinese import Rouge

from ais_bench.evaluate.common.global_var import *
from ais_bench.evaluate.common.log import logger

jieba.setLogLevel(logging.INFO)

EXCLUDE_LIST = ['.', ',', '。', '，', ' ', '(', ')', '"', "'"]
LEGAL_CHAR_PATTERN = r'^[\u4e00-\u9fa50-9a-zA-Z\s]+$'
EPSILON = 0.0001
BLEU_NGRAM_LIST = [1, 2, 3, 4]
ROUGE_TYPE_LIST = ["rouge-1", "rouge-2", "rouge-l"]
DISTINCT_NGRAM_LIST = [1, 2, 3, 4]

DEFAULT_THRESHOLD = {
    ACCURACY: 1, EDIT_DISTANCE: 5, BLEU: 0.4, ROUGE: 0.4,ABNORMAL_STRING_RATE: 0.3,
    DISTINCT: 0.6, RELATIVE_DISTINCT: 0.8,RELATIVE_ABNORMAL_STRING_RATE: 1.2
}

class BaseMeasurement(metaclass=ABCMeta):
    def __init__(self) -> None:
        self.metrics_list = []

    def clear(self):
        self.metrics_list = []

    def get_metrics_list(self):
        return self.metrics_list

    def filtering_helper(self, threshold):
        is_meet_threshold = self._meet_threshold(threshold)
        negative_index = []
        for i, metrics in enumerate(self.metrics_list):
            if not isinstance(metrics, (int, float)) or not is_meet_threshold(metrics):
                negative_index.append(i)
        return negative_index

    @abstractmethod
    def __call__(self, result_dict: dict, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def _meet_threshold(self, threshold=None): # return a function takes metrics as input return boolean
        raise NotImplementedError

def check_formated(result_dict):
    if not(isinstance(result_dict, dict) and TARGET_OUTPUT_INDEX in result_dict\
           and MODEL_OUTPUT_INDEX in result_dict and INDEX_INDEX in result_dict):
        logger.error(f"result_dict is not a dict or does not contains following keys"
                     f" {TARGET_OUTPUT_INDEX}, {MODEL_OUTPUT_INDEX}, {INDEX_INDEX}.")
        raise ValueError
    elif not (len(result_dict[TARGET_OUTPUT_INDEX]) == len(result_dict[MODEL_OUTPUT_INDEX])\
              == len(result_dict[INDEX_INDEX])):
        logger.error(f"the list in result_dict is not of the same length.")
        raise ValueError

def exclude(str_list, exclude_list):
    res_list = []
    for string in str_list:
        if string not in exclude_list:
            res_list.append(string)
    return res_list


class AccuracyMeasurement(BaseMeasurement):
    def __init__(self) -> None:
        super().__init__()
        self.name = ACCURACY

    def __call__(self, result_dict: dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_correct = 0
        for model_output, target_output in zip(result_dict[MODEL_OUTPUT_INDEX], result_dict[TARGET_OUTPUT_INDEX]):
            if self._post_process(model_output) == target_output:
                total_correct += 1
                self.metrics_list.append(1)
            else:
                self.metrics_list.append(0)
            total_amount += 1

        if total_amount == 0:
            accuracy = 0.0
        else:
            accuracy = total_correct / total_amount
        return accuracy

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
            threshold = DEFAULT_THRESHOLD[ACCURACY]
        def is_meet_threshold_function(metric):
            return metric >= threshold
        return is_meet_threshold_function


class EditDistanceMeasurement(BaseMeasurement):
    def __init__(self) -> None:
        super().__init__()
        self.name = EDIT_DISTANCE

    def __call__(self, result_dict: dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_edit_distance = 0

        for i, (model_output, target_output) in \
            enumerate(zip(result_dict[MODEL_OUTPUT_INDEX], result_dict[TARGET_OUTPUT_INDEX])):
            model_output = self._post_process(model_output)
            try:
                edit_distance = self._editing_distance(model_output, target_output)
            except Exception as e:
                logger.warning(f"case index: {i}, failed calculating metric: {self.name}."
                            f"exact exception: {e}.")
                self.metrics_list.append(None)
                continue
            total_edit_distance += edit_distance
            total_amount += 1
            self.metrics_list.append(edit_distance)

        if total_amount == 0:
            average_edit_distance = 0.0
        else:
            average_edit_distance = total_edit_distance / total_amount
        return average_edit_distance

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold):
            threshold = DEFAULT_THRESHOLD[EDIT_DISTANCE]
        def is_meet_threshold_function(metric):
            return metric <= threshold
        return is_meet_threshold_function

    def _editing_distance(self, word1, word2):
        dp = [[0] * (len(word2) + 1) for _ in range(len(word1) + 1)]
        for i in range(len(word1) + 1):
            dp[i][0] = i
        for j in range(len(word2) + 1):
            dp[0][j] = j
        for i in range(1, len(word1) + 1):
            for j in range(1, len(word2) + 1):
                if word1[i - 1] == word2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1]
                else:
                    dp[i][j] = min(dp[i - 1][j - 1], dp[i - 1][j], dp[i][j - 1]) + 1
        return dp[-1][-1]


class BleuMeasurement(BaseMeasurement):
    def __init__(self, ngram=1) -> None:
        super().__init__()
        if ngram not in BLEU_NGRAM_LIST:
            ngram = 1
            logger.warning(f"bleu ngram parameter is not one of {BLEU_NGRAM_LIST}, use default: {ngram}")
        weights_list = [0]*4
        weights_list[ngram-1] = 1
        self.weights = tuple(weights_list)
        self.name = BLEU + "-" + str(ngram)

    def __call__(self, result_dict:dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_bleu = 0

        for i, (model_output, target_output) in \
            enumerate(zip(result_dict[MODEL_OUTPUT_INDEX], result_dict[TARGET_OUTPUT_INDEX])):
            try:
                bleu = self._bleu_score(target_output, self._post_process(model_output))
            except Exception as e:
                logger.warning(f"case index: {i}, failed calculating metric: {self.name}."
                            f"exact exception: {e}.")
                self.metrics_list.append(None)
                continue
            total_amount += 1
            total_bleu += bleu
            self.metrics_list.append(bleu)

        if total_amount == 0:
            average_bleu = 0.0
        else:
            average_bleu = total_bleu / total_amount
        return average_bleu

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
            threshold = DEFAULT_THRESHOLD[BLEU]
        def is_meet_threshold_function(metric):
            return metric >= threshold
        return is_meet_threshold_function

    def _bleu_score(self, target_output, model_output):
        reference = list(jieba.cut(target_output))
        reference = exclude(reference, EXCLUDE_LIST)
        reference = [reference]

        candidate = list(jieba.cut(model_output))
        candidate = exclude(candidate, EXCLUDE_LIST)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bleu = bleu_score.sentence_bleu(reference, candidate, weights=self.weights)
        return bleu


class RougeMeasurement(BaseMeasurement):
    def __init__(self, rouge_type="rouge-1") -> None:
        super().__init__()
        if rouge_type in ROUGE_TYPE_LIST:
            self.rouge_type = rouge_type
        else:
            self.rouge_type = "rouge-1"
            logger.warning(f"rouge type is not one of {ROUGE_TYPE_LIST}, use default: {self.rouge_type}")
        self.name = self.rouge_type

    def __call__(self, result_dict:dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_rouge = 0

        for i, (model_output, target_output) in \
            enumerate(zip(result_dict[MODEL_OUTPUT_INDEX], result_dict[TARGET_OUTPUT_INDEX])):
            try:
                rouge = self._rouge(target_output, self._post_process(model_output))
            except Exception as e:
                logger.warning(f"case index: {i}, failed calculating metric: {self.name}."
                            f"exact exception: {e}.")
                self.metrics_list.append(None)
                continue
            total_amount += 1
            total_rouge += rouge
            self.metrics_list.append(rouge)

        if total_amount == 0:
            average_rouge = 0.0
        else:
            average_rouge = total_rouge / total_amount
        return average_rouge

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
            threshold = DEFAULT_THRESHOLD[ROUGE]
        def is_meet_threshold_function(metric):
            return metric >= threshold
        return is_meet_threshold_function

    def _rouge(self, target_output, model_output):
        rouge = Rouge()
        reference_list = list(jieba.cut(target_output))
        reference_list = exclude(reference_list, [" "])
        reference = ' '.join(reference_list)

        candidate_list = list(jieba.cut(model_output))
        candidate_list = exclude(candidate_list, [" "])
        candidate = ' '.join(candidate_list)

        scores = rouge.get_scores(candidate, reference)
        return scores[0][self.rouge_type]['f'] # retrun f-1 score


class DistinctMeasurement(BaseMeasurement):
    def __init__(self, ngram=2) -> None:
        super().__init__()
        if ngram in DISTINCT_NGRAM_LIST:
            self.ngram = ngram
        else:
            self.ngram = 2
            logger.warning(f"distinct ngram is not one of {DISTINCT_NGRAM_LIST}, use default: {self.ngram}")
        self.name = DISTINCT + '-' + str(self.ngram)

    def __call__(self, result_dict:dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_distinct = 0

        for i, model_output in enumerate(result_dict[MODEL_OUTPUT_INDEX]):
            try:
                distinct = self._distinct(self._post_process(model_output))
            except Exception as e:
                logger.warning(f"case index: {i}, failed calculating metric: {self.name}."
                            f"exact exception: {e}.")
                self.metrics_list.append(None)
                continue
            total_amount += 1
            total_distinct += distinct
            self.metrics_list.append(distinct)

        if total_amount == 0:
            average_distinct = 0.0
        else:
            average_distinct = total_distinct / total_amount
        return average_distinct

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
            threshold = DEFAULT_THRESHOLD[DISTINCT]
        def is_meet_threshold_function(metric):
            return metric >= threshold
        return is_meet_threshold_function

    def _distinct(self, output):
        words = list(jieba.cut(output))
        words = exclude(words, EXCLUDE_LIST)
        ngrams = [tuple(words[i: i + self.ngram]) for i in range(len(words) - self.ngram + 1)]

        total_count = len(ngrams)
        unique_count = len(Counter(ngrams))

        if total_count == 0:
            distinct_metric = 0
        else:
            distinct_metric = unique_count / total_count
        return distinct_metric


class RelativeDistinctMeasurement(BaseMeasurement):
    def __init__(self, ngram=2) -> None:
        super().__init__()
        if ngram in DISTINCT_NGRAM_LIST:
            self.ngram = ngram
        else:
            self.ngram = 2
            logger.warning(f"relative distinct ngram is not one of {DISTINCT_NGRAM_LIST}, use default: self.ngram")
        self.name = RELATIVE_DISTINCT + "-" + str(self.ngram)

    def __call__(self, result_dict:dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_relative_distinct = 0

        for i, (model_output, target_output) in \
            enumerate(zip(result_dict[MODEL_OUTPUT_INDEX], result_dict[TARGET_OUTPUT_INDEX])):
            try:
                relative_distinct = self._relative_distinct(self._post_process(model_output), target_output)
            except Exception as e:
                logger.warning(f"case index: {i}, failed calculating metric: {self.name}."
                            f"exact exception: {e}.")
                self.metrics_list.append(None)
                continue
            total_amount += 1
            total_relative_distinct += relative_distinct
            self.metrics_list.append(relative_distinct)

        if total_amount == 0:
            average_relative_distinct = 0.0
        else:
            average_relative_distinct = total_relative_distinct / total_amount
        return average_relative_distinct

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold):
            threshold = DEFAULT_THRESHOLD[RELATIVE_DISTINCT]
        def is_meet_threshold_function(metric):
            return metric >= threshold
        return is_meet_threshold_function

    def _distinct(self, output):
        words = list(jieba.cut(output))
        words = exclude(words, EXCLUDE_LIST)
        ngrams = [tuple(words[i: i + self.ngram]) for i in range(len(words) - self.ngram + 1)]

        total_count = len(ngrams)
        unique_count = len(Counter(ngrams))

        if total_count == 0:
            distinct_metric = 0.0
        else:
            distinct_metric = unique_count / total_count
        return distinct_metric

    def _relative_distinct(self, model_output, target_output):
        distinct_model = self._distinct(model_output)
        distinct_target = self._distinct(target_output)
        if distinct_target == 0.0:
            distinct_target = EPSILON
        return distinct_model / distinct_target


class AbnormalStringRateMeasurement(BaseMeasurement):
    def __init__(self) -> None:
        super().__init__()
        self.name = ABNORMAL_STRING_RATE

    def __call__(self, result_dict:dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_abnormal_string_rate = 0

        for i, model_output in enumerate(result_dict[MODEL_OUTPUT_INDEX]):
            try:
                abnormal_string_rate = self._abnormal_string_rate(self._post_process(model_output))
            except Exception as e:
                logger.warning(f"case index: {i}, failed calculating metric: {self.name}."
                            f"exact exception: {e}.")
                self.metrics_list.append(None)
                continue
            total_amount += 1
            total_abnormal_string_rate += abnormal_string_rate
            self.metrics_list.append(abnormal_string_rate)

        if total_amount == 0:
            average_abnormal_string_rate = 0.0
        else:
            average_abnormal_string_rate = total_abnormal_string_rate / total_amount
        return average_abnormal_string_rate

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold <= 1):
            threshold = DEFAULT_THRESHOLD[ABNORMAL_STRING_RATE]
        def is_meet_threshold_function(metric):
            return metric <= threshold
        return is_meet_threshold_function

    def _abnormal_string_rate(self, model_output):
        pattern = LEGAL_CHAR_PATTERN
        words = list(jieba.cut(model_output))
        words = exclude(words, EXCLUDE_LIST)
        total_amount = len(words)
        total_abnormal = 0
        if total_amount == 0:
            abnormal_string_rate = 0.0
        else:
            for word in words:
                if not re.match(pattern, word):
                    total_abnormal += 1
            abnormal_string_rate = total_abnormal / total_amount
        return abnormal_string_rate


class RelativeAbnormalStringRateMeasurement(BaseMeasurement):
    def __init__(self) -> None:
        super().__init__()
        self.name = RELATIVE_ABNORMAL_STRING_RATE

    def __call__(self, result_dict:dict, **kwargs):
        check_formated(result_dict)
        self.clear()

        total_amount = 0
        total_relative_abnormal_string_rate = 0

        for i, (model_output, target_output) in \
            enumerate(zip(result_dict[MODEL_OUTPUT_INDEX], result_dict[TARGET_OUTPUT_INDEX])):
            try:
                relative_abnormal_string_rate = \
                    self._relative_abnormal_string_rate(self._post_process(model_output), target_output)
            except Exception as e:
                logger.warning(f"case index: {i}, failed calculating metric: {self.name}."
                            f"exact exception: {e}.")
                self.metrics_list.append(None)
                continue

            total_amount += 1
            total_relative_abnormal_string_rate += relative_abnormal_string_rate
            self.metrics_list.append(relative_abnormal_string_rate)

        if total_amount == 0:
            average_relative_abnormal_string_rate = 0.0
        else:
            average_relative_abnormal_string_rate = total_relative_abnormal_string_rate / total_amount
        return average_relative_abnormal_string_rate

    def _post_process(self, model_output):
        return model_output

    def _meet_threshold(self, threshold=None):
        if not (isinstance(threshold, (int, float)) and 0 <= threshold):
            threshold = DEFAULT_THRESHOLD[RELATIVE_ABNORMAL_STRING_RATE]
        def is_meet_threshold_function(metric):
            return metric <= threshold
        return is_meet_threshold_function

    def _abnormal_string_rate(self, model_output):
        pattern = LEGAL_CHAR_PATTERN
        words = list(jieba.cut(model_output))
        words = exclude(words, EXCLUDE_LIST)
        total_amount = len(words)
        total_abnormal = 0
        if total_amount == 0:
            abnormal_string_rate = 0.0
        else:
            for word in words:
                if not re.match(pattern, word):
                    total_abnormal += 1
            abnormal_string_rate = total_abnormal / total_amount
        return abnormal_string_rate

    def _relative_abnormal_string_rate(self, model_output, target_output):
        abnormal_string_rate_model = self._abnormal_string_rate(model_output)
        abnormal_string_rate_target = self._abnormal_string_rate(target_output)
        if abnormal_string_rate_target == 0.0:
            abnormal_string_rate_target = EPSILON
        return abnormal_string_rate_model / abnormal_string_rate_target