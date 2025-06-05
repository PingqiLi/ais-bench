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
class VideoBenchDataset(BaseDataset):

    @staticmethod
    def load(path):
        path = get_data_path(path, local_mode=True)
        ans_path = path + '/answer/ANSWER.json'
        if not os.path.exists(ans_path):
            raise FileNotFoundError("Cannot find answer file, Please check your datasets!")
        with open(ans_path, 'r', encoding='utf-8') as f_ans:
            answers = json.load(f_ans)
        path = Path(path)
        dataset = []
        for sub_path in path.glob("*new.json"):
            with open(sub_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            for key in data.keys():
                try:
                    dataset_name = data[key]['vid_path'].split('/')[-2]
                    answer = answers[dataset_name][key]
                    dataset.append({"vid_path": data[key]["vid_path"],
                                    "video_id": str(data[key]["video_id"]),
                                    "question": data[key]["question"],
                                    "choices": data[key]["choices"],
                                    'answer': answer})
                except:
                    raise ValueError("Please check your datasets!")
                
        return Dataset.from_list(dataset)


class VideoBenchEvaluator(BaseEvaluator):

    def find_choice(self, result):
        choice_list = ['A', 'B', 'C', 'D', 'E', 'F']
        for choice in choice_list:
            if choice in result:
                return choice
        return ""

    def score(self, predictions, references):
        references = [i['answer'] for i in references]
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
            if self.find_choice(i) == j:
                correct += 1
                detail['correct'] = True
            details.append(detail)
        result = {'accuracy': 100 * correct / count, 'details': details}
        return result