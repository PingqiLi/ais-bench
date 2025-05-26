import json
import os
import re
from os import environ
from pathlib import Path

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.openicl import BaseEvaluator
from ais_bench.benchmark.registry import LOAD_DATASET, TEXT_POSTPROCESSORS
from ais_bench.benchmark.utils import get_data_path

from .base import BaseDataset


@LOAD_DATASET.register_module()
class VocalSoundDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        path = Path(path)
        dataset = []
        for file_path in path.glob("*.wav"):
            try:
                answer = str(file_path).split('.')[0].split('_')[-1]
                dataset.append({"audio_path": str(file_path),
                                "question": "To be replaced!",
                                'answer': answer})
            except:
                raise ValueError("Please check your datasets!")
                
        return Dataset.from_list(dataset)


class VocalSoundEvaluator(BaseEvaluator):

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