import json
import os
import re
from os import environ

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class TEXTVQADataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        dataset = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = json.loads(line.strip())
                dataset.append(line)
        return Dataset.from_list(dataset)


class TEXTEvaluator(BaseEvaluator):

    def is_equal(self, pred, refer):
        try:
            if pred == refer or abs(float(pred) - int(refer)) < 1e-6:
                return True
        except Exception:
            pass
        return False

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        correct = 0
        count = 0
        details = []
        for i, j in zip(predictions, references):
            detail = {'pred': i, 'answer': j, 'correct': False}
            count += 1
            if self.is_equal(i, j):
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result