import csv
import json
import mmap
import os.path as osp
from collections import defaultdict
from typing import Dict, List

import orjson
import tabulate
from mmengine import ConfigDict

from ais_bench.benchmark.calculators.base_perf_metric_calculator import BasePerfMetricCalculator
from ais_bench.benchmark.utils import (
    get_logger,
    model_abbr_from_cfg,
    plot_sorted_request_timelines,
)
from ais_bench.benchmark.utils.build import build_perf_metric_calculator_from_cfg
from ais_bench.benchmark.utils.results import dump_results_dict, load_from_h5


def model_abbr_from_cfg_used_in_summarizer(model):
    """Get model abbreviation for summarizer.

    Args:
        model: Model configuration dictionary

    Returns:
        str: Model abbreviation
    """
    if model.get("summarizer_abbr", None):
        return model["summarizer_abbr"]
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

        self.model_cfgs = self.cfg["models"]
        self.dataset_cfgs = self.cfg["datasets"]
        self.merge_ds = self.cfg.get("cli_args", {}).get("merge_ds", False)
        self.calculator_conf = calculator
        self.calculators = {}

        dataset_abbrs = []
        merge_ds_abbrs = defaultdict(list)
        if self.merge_ds:
            # In merge_ds mode, group datasets by type for performance results
            for dataset_cfg in self.dataset_cfgs:
                merged_ds_abbr = dataset_cfg.get("type").split(".")[-1].lower()
                merge_ds_abbrs[merged_ds_abbr].append(dataset_cfg.get("abbr"))
            for ds_abbrs in merge_ds_abbrs.values():
                dataset_abbrs.append(ds_abbrs)
        else:
            # Index *_details.jsonl files by dataset abbreviation
            for dataset_cfg in self.dataset_cfgs:
                dataset_abbrs.append([dataset_cfg.get("abbr")])
        self.dataset_abbrs = dataset_abbrs

        self.work_dir = self.cfg["work_dir"]
        model_abbrs = []
        for model in self.model_cfgs:
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model)
            if model_abbr in model_abbrs:
                continue
            model_abbrs.append(model_abbr)
        self.model_abbrs = model_abbrs

    def _get_dataset_abbr(self, dataset_abbrs):
        """Get dataset abbreviation.

        Args:
            dataset_abbrs: List of dataset abbreviations

        Returns:
            str: Dataset abbreviation
        """
        return (
            dataset_abbrs[0].get("type").split(".")[-1].lower()
            if self.merge_ds
            else dataset_abbrs[0]
        )

    def _extract_success_cases(self, details_perf_datas):
        """Extract successful cases from performance data.

        Performance calculation only keeps successful results, success is used to count failures.

        Args:
            details_perf_datas: Performance data dictionary

        Returns:
            dict: Performance data with only successful cases
        """
        success_id = [i for i, ok in enumerate(details_perf_datas["success"]) if ok]
        success_details_perf_datas = {
            key: [value[i] for i in success_id]
            for key, value in details_perf_datas.items()
            if key != "success"
        }
        success_details_perf_datas["success"] = details_perf_datas["success"]
        return success_details_perf_datas

    def _load_details_perf_data(self, model_abbr, data_abbrs):
        """Load details performance data and h5 data based on data_abbrs.

        Maps h5 data back to details data.

        Args:
            model_abbr: Model abbreviation
            data_abbrs: List of data abbreviations

        Returns:
            dict: Details performance data
        """
        perf_datas = []
        h5_names = []

        for data_abbr in data_abbrs:
            perf_details_file = osp.join(
                self.work_dir, "performances", model_abbr, f"{data_abbr}_details.jsonl"
            )
            if not osp.exists(perf_details_file):
                self.logger.warning(
                    f"Cannot find {data_abbr} details perf data in {perf_details_file}, skip."
                )
                continue
            with open(perf_details_file, "rb") as f:
                mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                for line in iter(mm.readline, b""):
                    perf_datas.append(orjson.loads(line))
                    h5_name = list(perf_datas[-1].values())[0].get("h5_name")
                    if h5_name and h5_name not in h5_names:
                        h5_names.append(h5_name)

        array_data = {}
        for h5_name in h5_names:
            h5_file = osp.join(
                self.work_dir, "performances", model_abbr, "h5_data", h5_name
            )
            h5_data = load_from_h5(h5_file)
            array_data.update(h5_data)
        details_perf_datas = defaultdict(list)

        def recursive_update(pre_key, detail_data):
            """Recursively update performance data.

            Args:
                pre_key: Previous key
                detail_data: Detail data to process
            """
            if not detail_data:
                details_perf_datas[pre_key].append(detail_data)
                return
            if isinstance(detail_data, dict):
                # __h5_ref__ marks ndarray data
                if "__h5_ref__" in detail_data:
                    details_perf_datas[pre_key].append(
                        array_data[detail_data["__h5_ref__"]]
                    )
                    return
                for key, value in detail_data.items():
                    recursive_update(key, value)
            else:
                details_perf_datas[pre_key].append(detail_data)

        for perf_data in perf_datas:
            for data_id, detail_data in perf_data.items():
                detail_data["id"] = data_id
                recursive_update(data_id, detail_data)
        lens = {key: len(value) for key, value in details_perf_datas.items()}
        if len(set(list(lens.values()))) != 1:
            raise ValueError(
                f"The length of details perf datas is not the same: {lens}, "
                f"each perf data should have same data structure"
            )
        details_perf_datas = self._extract_success_cases(details_perf_datas)
        return details_perf_datas

    def _dump_calculated_perf_data(self):
        """Dump calculated performance data to files.

        Saves both JSON and CSV formats for each model and dataset combination.
        """
        for model, calc_per_ds in self.calculators.items():
            for dataset, calc in calc_per_ds.items():
                calc.calculate()
                output_filepath = osp.join(self.work_dir, "performances", model)
                dump_results_dict(
                    calc.get_common_res(),
                    osp.join(output_filepath, dataset + ".json"),
                )
                calc.save_performance(osp.join(output_filepath, dataset + ".csv"))

    def _pick_up_results(self):
        """Pick up performance results from files.

        Returns:
            Dict[str, List]: Performance tables dictionary
        """
        # perf_tables: {"model_abbr/dataset_abbr": result_table}
        perf_tables: Dict[str, List] = {}
        for model in self.model_abbrs:
            for dataset_abbrs in self.dataset_abbrs:
                dataset_abbr = self._get_dataset_abbr(dataset_abbrs)
                perf_result_dir = osp.join(self.work_dir, "performances", model)
                table_list = []
                if osp.exists(osp.join(perf_result_dir, f"{dataset_abbr}.csv")):
                    table_list.append(
                        self._load_csv_to_table(
                            osp.join(perf_result_dir, f"{dataset_abbr}.csv")
                        )
                    )
                if osp.exists(osp.join(perf_result_dir, f"{dataset_abbr}.json")):
                    table_list.append(
                        self._load_json_to_table(
                            osp.join(perf_result_dir, f"{dataset_abbr}.json")
                        )
                    )
                else:
                    self.logger.warning(
                        f"Cannot find {dataset_abbr} common performance results in {perf_result_dir}, skip."
                    )
                perf_tables[f"{model}/{dataset_abbr}"] = table_list

        return perf_tables

    def _load_csv_to_table(self, csv_path):
        """Load CSV file and convert to table format.

        Args:
            csv_path: Path to CSV file

        Returns:
            List[List]: Table data
        """
        table = []
        with open(csv_path, "r", newline="", encoding="utf-8") as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                table.append(row)
        return table

    def _load_json_to_table(self, json_path):
        """Load JSON file and convert to table format.

        Args:
            json_path: Path to JSON file

        Returns:
            List[List]: Table data
        """
        table = [["Common Metric", "Stage", "Value"]]
        with open(json_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        for key, stage_value in data.items():
            for stage_name, value in stage_value.items():
                table.append([key, stage_name, value])
        return table

    def _output_to_screen(self, tables_dict: Dict):
        """Output performance results to screen.

        Args:
            tables_dict: Dictionary containing performance tables
        """
        for task_name, tables in tables_dict.items():
            self.logger.info(f"Performance Results of task: {task_name}: ")
            for table in tables:
                print(
                    tabulate.tabulate(
                        table,
                        headers="firstrow",
                        tablefmt="fancy_grid",  # Use bordered table style
                        floatfmt=".2f",  # Keep two decimal places
                        numalign="center",  # Center align numbers
                        stralign="left",  # Left align text
                        missingval="N/A",  # Handle empty values
                    )
                )
            model_name = task_name.split("/")[0]
            perf_result_dir = osp.join(self.work_dir, "performances", model_name)
            self.logger.info(f"Performance Result files located in {perf_result_dir}.")

    def summarize(self):
        """Summarize performance results for all models and datasets.

        Processes service models, calculates performance metrics, generates plots,
        and outputs results to screen and files.
        """
        for model_cfg in self.model_cfgs:
            if not model_cfg.get("attr") == "service":
                continue
            model_abbr = model_abbr_from_cfg_used_in_summarizer(model_cfg)
            max_concurrency = model_cfg.get("batch_size", 1)
            calculators_per_model = {}
            for dataset_abbrs in self.dataset_abbrs:
                details_perf_datas = self._load_details_perf_data(
                    model_abbr, dataset_abbrs
                )
                # In merge_ds mode, use datatype of similar datasets as abbreviation
                dataset_abbr = self._get_dataset_abbr(dataset_abbrs)
                # Generate visualization HTML file
                plot_file_path = osp.join(
                    self.work_dir,
                    "performances",
                    model_abbr,
                    f"{dataset_abbr}_plot.html",
                )
                has_plot = plot_sorted_request_timelines(
                    details_perf_datas["start_time"],
                    details_perf_datas["ttft"],
                    details_perf_datas["end_time"],
                    details_perf_datas["itl"],
                    details_perf_datas.get(
                        "multiturn_group_id",
                        [""] * len(details_perf_datas["start_time"]),
                    ),
                    output_file=plot_file_path,
                    unit="s",
                )
                if has_plot:
                    self.logger.info(
                        f"The {dataset_abbr}_plot has been saved in {plot_file_path}"
                    )
                calculator: BasePerfMetricCalculator = (
                    build_perf_metric_calculator_from_cfg(self.calculator_conf)
                )
                calculators_per_model[dataset_abbr] = calculator
                try:
                    calculator._init_datas(details_perf_datas, max_concurrency)
                except RuntimeError as e:
                    self.logger.error(
                        f'Failed to calculate performance data, detail error is: "{e}", '
                        f"please check {plot_file_path} to do further analysis."
                    )
                    raise RuntimeError("Calculate perf data failed!")

            self.calculators[model_abbr] = calculators_per_model
        self._dump_calculated_perf_data()
        # Pick up results
        perf_tables = self._pick_up_results()

        # Output to screen
        self._output_to_screen(perf_tables)
