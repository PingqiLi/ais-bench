import copy
import csv
import json
import os
from typing import List

from datasets import Dataset

from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator, BaseEvaluator
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.registry import LOAD_DATASET
from ais_bench.benchmark.utils import get_data_path
from ais_bench.benchmark.utils.datasets import get_sample_data, safe_load_json_file, safe_load_csv_file
from ais_bench.benchmark.utils.types import check_meta_json_dict

from .base import BaseDataset


class OptionSimAccEvaluator(BaseEvaluator):

    def __init__(self, options) -> None:
        super().__init__()
        if not all((isinstance(i, str) and i.isupper() and len(i) == 1)
                   for i in options):
            raise ValueError(
                f'Each options should be single upper letter, got {options}')

        self.options = options

    def match_any_label(self, pred, test_item):
        from rapidfuzz.distance import Levenshtein as L

        from ais_bench.benchmark.utils.text_postprocessors import \
            first_option_postprocess

        pred = pred.strip()
        if any([pred == i for i in self.options]):
            parsed = pred
        else:
            parsed = ''
        if parsed == '':
            parsed = first_option_postprocess(pred,
                                              ''.join(self.options),
                                              cushion=False)
        if parsed == '':
            possible_options = []
            for opt in self.options:
                opt_str = test_item[opt]
                if opt_str is not None and opt_str.lower() in pred.lower():
                    possible_options.append(opt)
            if len(possible_options) == 1:
                parsed = possible_options[0]
        if parsed == '':
            dists = []
            for opt in self.options:
                opt_str = test_item[opt]
                if opt_str is None:
                    continue
                cands = [opt, opt_str, opt + '. ' + opt_str]
                d = min(L.distance(pred, cand) for cand in cands)
                dists.append((d, opt))
            if len(dists) > 0:
                parsed = min(dists)[1]
        return parsed

    def score(self, predictions: List, references: List, test_set) -> dict:
        assert len(predictions) == len(references)

        num_correct, num_total = 0, 0
        details = {}
        for index in range(len(predictions)):
            pred = predictions[index]
            refr = references[index]
            parsed = self.match_any_label(pred, test_set[index])
            num_correct += 1 if parsed == refr else 0
            num_total += 1
            details[str(index)] = {}
            details[str(index)]['pred'] = pred
            details[str(index)]['parsed'] = parsed
            details[str(index)]['refr'] = refr
            details[str(index)]['correct'] = parsed == refr
        return {'accuracy': num_correct / num_total * 100, 'details': details}


@LOAD_DATASET.register_module()
class CustomDataset(BaseDataset):

    @staticmethod
    def load(path, file_name=None, local_mode=False, meta_json_conf=None):
        path = get_data_path(path, local_mode=True)
        if file_name is not None:
            path = os.path.join(path, file_name)
        if path.endswith('.jsonl'):
            data = safe_load_json_file(path, expected_types=(dict))
        elif path.endswith('.csv'):
            data = safe_load_csv_file(path, expected_types=(str))
        else:
            raise ValueError(f'Unsupported file format: {path}')
        if meta_json_conf is not None:
            meta_json_conf = check_meta_json_dict(meta_json_conf)
            sample_mode = meta_json_conf.get('sampling_mode', 'default')
            request_count = meta_json_conf.get('request_count', 0)
            sample_data = get_sample_data(data, sample_mode, int(request_count))
            return Dataset.from_list(sample_data)
        return Dataset.from_list(data)


def stringfy_types(obj):
    for k, v in obj.items():
        if k == 'type':
            obj[k] = f'{v.__module__}.{v.__name__}'
        elif isinstance(v, dict):
            stringfy_types(v)
    return obj


def make_mcq_gen_config(meta):
    if meta.get('template', None) is None:
        _human_prompt = 'Question: {question}' + ''.join(
            [f'\n{item}. {{{item}}}' for item in meta['options']])
        human_prompt = meta.get('human_prompt', _human_prompt)
        _bot_prompt = f'Answer: {{{meta["output_column"]}}}'
        bot_prompt = meta.get('bot_prompt', _bot_prompt)
        template = human_prompt + "\n" + bot_prompt
    else:
        template = meta['template']

    reader_cfg = dict(
        input_columns=meta['input_columns'],
        output_column=meta['output_column'],
    )
    if max_tokens_column:= meta.get('max_tokens_column'):
        reader_cfg.update({'max_tokens_column':max_tokens_column})
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator', OptionSimAccEvaluator),
                       **meta.get('evaluator_kwargs',
                                  {'options': meta['options']})),
        pred_role='BOT',
    )

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        meta_json_conf=meta['meta_json_conf'],
    )
    return dataset


def make_qa_gen_config(meta):
    if meta.get('template', None) is None:
        human_prompt = meta.get('human_prompt', '{question}')
        if meta['output_column'] is None:
            template = human_prompt,
        else:
            bot_prompt = meta.get('bot_prompt', f'{{{meta["output_column"]}}}')
            template = "Question: "+ human_prompt + "\nAnswer: " + bot_prompt
    else:
        template = meta['template']
    reader_cfg = dict(
        input_columns=meta['input_columns'],
        output_column=meta['output_column'],
    )
    if max_tokens_column:= meta.get('max_tokens_column'):
        reader_cfg.update({'max_tokens_column':max_tokens_column})
    if 'test_range' in meta:
        reader_cfg['test_range'] = meta['test_range']
    infer_cfg = dict(
        prompt_template=dict(
            type=PromptTemplate,
            template=template,
        ),
        retriever=dict(type=ZeroRetriever),
        inferencer=dict(type=GenInferencer),
    )

    eval_cfg = dict(
        evaluator=dict(type=meta.get('evaluator', AccEvaluator),
                       **meta.get('evaluator_kwargs', {})),
        pred_role='BOT',
    )

    dataset = dict(
        abbr=meta['abbr'],
        type=CustomDataset,
        path=meta['path'],
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
        meta_json_conf=meta['meta_json_conf'],
    )
    return dataset


def parse_example_dataset(config):
    # config -> .meta.jsonl -> parsed_results
    path = config['path']

    # load sample and get parsed_meta
    parsed_meta = {}
    if path.endswith('.jsonl'):
        with open(path, 'r', encoding='utf-8') as f:
            data_item = json.loads(f.readline())
    elif path.endswith('.csv'):
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            row = next(reader)
            data_item = dict(zip(header, row))
    else:
        raise ValueError(f'Unsupported ext: {path}, .jsonl or .csv required')

    parsed_meta['path'] = path

    input_columns = [i for i in data_item.keys() if i == 'question']
    if not input_columns:
        raise ValueError(f'Unsupported dataset format: NOT contain "question" in {path}')
    parsed_meta['input_columns'] = input_columns
    output_column = 'answer' if 'answer' in data_item else None
    parsed_meta['output_column'] = output_column
    max_tokens_column = 'max_tokens' if 'max_tokens' in data_item else None
    parsed_meta['max_tokens_column'] = max_tokens_column

    options = []
    for i in range(26):
        i = chr(ord('A') + i)
        if i in data_item:
            options.append(i)
        else:
            break
    parsed_meta['options'] = options
    abbr = os.path.basename(path).split('.')[0]
    parsed_meta['abbr'] = abbr
    parsed_meta['data_type'] = 'mcq' if len(options) > 1 else 'qa'
    parsed_meta['infer_method'] = 'gen'

    # try to read meta json
    meta_path = config.get('meta_path', path + '.meta.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            read_from_file_meta = json.load(f)
    else:
        if "meta_path" in config: 
            # user set custom-dataset-meta-path does not exists
            raise ValueError(f'The file path specified by parameter "custom-dataset-meta-path" does not exist: {meta_path}')
        read_from_file_meta = {}
    parsed_meta['meta_json_conf'] = read_from_file_meta

    # get config meta
    config_meta = copy.deepcopy(config)

    # merge meta
    meta = {}
    meta.update(parsed_meta)
    meta.update(config_meta)

    return meta


def make_custom_dataset_config(config):
    # considered as a custom dataset
    meta = parse_example_dataset(config)
    make_config_func = {
        ('mcq', 'gen'): make_mcq_gen_config,
        ('qa', 'gen'): make_qa_gen_config,
    }.get((meta['data_type'], meta['infer_method']), None)
    if make_config_func is None:
        raise ValueError(f'Unsupported dataset data_type: {meta["data_type"]}'
                         f' and infer_method: {meta["infer_method"]}')
    dataset = make_config_func(meta)
    dataset = stringfy_types(dataset)
    return dataset
