# flake8: noqa
# yapf: disable
import functools
import getpass
import math
import csv
import json
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, Optional

import mmengine
import tabulate
from mmengine import ConfigDict

from ais_bench.benchmark.utils import (LarkReporter, dataset_abbr_from_cfg, get_infer_merged_output_path,
                               get_infer_output_path, get_logger, merged_dataset_abbr_from_class,
                               model_abbr_from_cfg)
from ais_bench.benchmark.utils.prompt import get_prompt_hash

METRIC_WHITELIST = ['score', 'auc_score', 'accuracy', 'humaneval_pass@1', 'rouge1', 'avg_toxicity_score', 'bleurt_diff', 'matthews_correlation', 'truth', 'f1', 'exact_match', 'extract_rate']
METRIC_BLACKLIST = ['bp', 'sys_len', 'ref_len', 'type']

def model_abbr_from_cfg_used_in_summarizer(model):
    if model.get('summarizer_abbr', None):
        return model['summarizer_abbr']
    else:
        return model_abbr_from_cfg(model)


class DefaultPerfSummarizer:
    """Default summarizer in AISBench.

    Args:
        config (ConfigDict): The configuration object of the evaluation task. It's expected to be filled out at runtime.
        dataset_abbrs (list[str], optional): Dataset abbreviations to be listed in the summary.
        summary_groups (list): The dataset groups whose results need to be averaged out. For example, mmlu. Each item it a dict with
            'name' (str) and 'subsets' (list of dataset abbrs), and optionally
            'weights' if weighted average is needed.
        prompt_db: A deprecated field.
    """

    def __init__(self, config: ConfigDict) -> None:
        self.tasks = []
        self.cfg = config
        self.logger = get_logger()

        self.model_cfgs = self.cfg['models']
        self.dataset_cfgs = self.cfg['datasets']

        dataset_abbrs = []
        for dataset_cfg in self.dataset_cfgs:
            merged_ds_abbr = dataset_cfg.get('type').split('.')[-1].lower()
            if merged_ds_abbr not in dataset_abbrs :
                dataset_abbrs.append(merged_ds_abbr)
        self.dataset_abbrs = dataset_abbrs

        self.work_dir = self.cfg['work_dir']
        model_abbrs = []
        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            if model_abbr in model_abbrs:
                continue
            model_abbrs.append(model_abbr)
        self.model_abbrs = model_abbrs

    def _pick_up_results(self):

        # perf_tables: {"model_abbr/dataset_abbr": result_table}
        perf_tables : Dict[str, []] = {}

        for model in self.model_abbrs:
            for dataset in self.dataset_abbrs:
                perf_result_dir = osp.join(self.work_dir, "performances", model)
                if osp.exists(osp.join(perf_result_dir, f"{dataset}.csv")):
                    perf_tables[f"{model}/{dataset}"] = self._load_csv_to_table(osp.join(perf_result_dir, f"{dataset}.csv"))
                elif osp.exists(osp.join(perf_result_dir, f"{dataset}.json")):
                    perf_tables[f"{model}/{dataset}"] = self._load_json_to_table(osp.join(perf_result_dir, f"{dataset}.json"))
                else:
                    self.logger.warning(f"Can not find {dataset} performance results in {perf_result_dir}, skip.")

        return perf_tables

    def _load_csv_to_table(self, csv_path):
        table = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                table.append(row)
        return table

    def _load_json_to_table(self, json_path):
        table = [["Performance Parameters", "Average"]]
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for key, value in data.items():
            table.append([key, value])
        return table

    def _output_to_screen(self, tables_dict: Dict):
        for task_name, table in tables_dict.items():
            self.logger.info(f"Performance Results of task: {task_name}: ")
            print(
                tabulate.tabulate(
                    table,
                    headers='firstrow',
                    tablefmt="fancy_grid",  # 使用带边框的表格样式
                    floatfmt=".2f",         # 保留两位小数
                    numalign="center",      # 数字列居中对齐
                    stralign="left",        # 文本列左对齐
                    missingval="N/A",       # 处理空值
                )
            )
            model_name = task_name.split("/")[0]
            perf_result_dir = osp.join(self.work_dir, "performances", model_name)
            self.logger.info(f"Performance Result files locate in {perf_result_dir}.")

    def summarize(self):  # noqa
        # pick up results
        perf_tables = self._pick_up_results()

        # output to screen
        self._output_to_screen(perf_tables)
