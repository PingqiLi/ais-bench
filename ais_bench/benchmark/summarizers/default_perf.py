# flake8: noqa
# yapf: disable
import functools
import getpass
import math
import csv
import json
import orjson
import os.path as osp
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

import mmengine
import tabulate
from mmengine import ConfigDict

from ais_bench.benchmark.utils import (LarkReporter, dataset_abbr_from_cfg, get_infer_merged_output_path,
                               get_infer_output_path, get_logger, merged_dataset_abbr_from_class,
                               model_abbr_from_cfg, plot_sorted_request_timelines)
from ais_bench.benchmark.utils.prompt import get_prompt_hash
from ais_bench.benchmark.utils.build import build_perf_metric_calculator_from_cfg
from ais_bench.benchmark.utils.results import dump_results_dict, load_from_h5


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

    def __init__(self, config: ConfigDict, calculator: ConfigDict) -> None:
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
        if self.model_cfgs[0].get("attr") == "service":
            self._load_details_perf_data(calculator)
            self._dump_calculated_perf_data()

    def extract_success_item(self, requests: dict):
        is_success = requests.get("is_success", [])
        for key, value in requests.items():
            if key == "is_success":
                continue
            if isinstance(value, list):
                value = [v for i, v in enumerate(value) if is_success[i]]
            else:
                value = value if is_success else []
            requests[key] = value
        return requests

    def _load_details_perf_data(self, calculator_conf: ConfigDict):
        self.calculators = {}
        for model in self.model_abbrs:
            calculators_per_model = {}
            for dataset in self.dataset_abbrs:
                perf_details_file = osp.join(self.work_dir, "performances", model, f"{dataset}_details.json")
                if not osp.exists(perf_details_file):
                    continue
                self.logger.info(f"Loading detail perf data of {model=} {dataset=} ...")
                details_data = orjson.loads(open(perf_details_file, "rb").read())
                details_data["requests"] = self.extract_success_item(details_data.get("requests", {}))
                is_success = details_data["requests"].get("is_success", [])
                decode_cost_file = osp.join(self.work_dir, "performances", model, f"{dataset}_details.h5")
                h5_data = load_from_h5(decode_cost_file)
                details_data["requests"]["decode_token_latencies"] = [value for i, value in enumerate(h5_data) if is_success[i]]
                plot_file_path = osp.join(self.work_dir, "performances", model, f"{dataset}_plot.html")
                has_plot = plot_sorted_request_timelines(
                    details_data["requests"]["start_time"],
                    details_data["requests"]["prefill_latency"],
                    details_data["requests"]["end_time"],
                    details_data["requests"]["decode_token_latencies"],
                    details_data["requests"]["multiturn_group_id"],
                    output_file=plot_file_path, unit="s"
                )
                if has_plot:
                    self.logger.info(f"The {dataset}_plot has been saved in {plot_file_path}")
                calculators_per_model[dataset] = build_perf_metric_calculator_from_cfg(calculator_conf)
                try:
                    calculators_per_model[dataset]._init_datas(details_data)
                except RuntimeError as e:
                    self.logger.error(f"Failed to calculate performance data, detail error is: \"{e}\", please check {plot_file_path} to do further analysis.")
                    raise RuntimeError("Calculate perf data failed!")

            self.calculators[model] = calculators_per_model

    def _dump_calculated_perf_data(self):
        for model, calc_per_ds in self.calculators.items():
            for dataset, calc in calc_per_ds.items():
                calc.calculate()
                output_filepath = osp.join(self.work_dir, "performances", model)
                dump_results_dict(
                    calc.get_common_res(),
                    osp.join(output_filepath, dataset + ".json"),
                )
                calc.save_performance(
                    osp.join(output_filepath, dataset + ".csv")
                )

    def _pick_up_results(self):

        # perf_tables: {"model_abbr/dataset_abbr": result_table}
        perf_tables : Dict[str, []] = {}

        for model in self.model_abbrs:
            for dataset in self.dataset_abbrs:
                perf_result_dir = osp.join(self.work_dir, "performances", model)
                table_list = []
                if osp.exists(osp.join(perf_result_dir, f"{dataset}.csv")):
                    table_list.append(self._load_csv_to_table(osp.join(perf_result_dir, f"{dataset}.csv")))
                if osp.exists(osp.join(perf_result_dir, f"{dataset}.json")):
                    table_list.append(self._load_json_to_table(osp.join(perf_result_dir, f"{dataset}.json")))
                else:
                    self.logger.warning(f"Can not find {dataset} common performance results in {perf_result_dir}, skip.")
                perf_tables[f"{model}/{dataset}"] = table_list

        return perf_tables

    def _load_csv_to_table(self, csv_path):
        table = []
        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                table.append(row)
        return table

    def _load_json_to_table(self, json_path):
        table = [["Common Metric", "Stage", "Value"]]
        with open(json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        for key, stage_value in data.items():
            for stage_name, value in stage_value.items():
                table.append([key, stage_name, value])
        return table

    def _output_to_screen(self, tables_dict: Dict):
        for task_name, tables in tables_dict.items():
            self.logger.info(f"Performance Results of task: {task_name}: ")
            for table in tables:
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
