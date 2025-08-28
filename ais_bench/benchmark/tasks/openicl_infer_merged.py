import argparse
import os
import os.path as osp
import random
import sys
import time
from typing import Any

from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import (
    ICL_INFERENCERS,
    ICL_PROMPT_TEMPLATES,
    ICL_RETRIEVERS,
    TASKS,
)
from ais_bench.benchmark.tasks.base import BaseTask
from ais_bench.benchmark.utils import (
    build_dataset_from_cfg,
    build_model_from_cfg,
    get_infer_output_path,
    get_perf_output_path,
    get_infer_merged_output_path,
    get_logger,
    model_abbr_from_cfg,
    task_abbr_from_cfg,
)
from ais_bench.benchmark.utils.types import _check_type


@TASKS.register_module()
class OpenICLInferMergedTask(BaseTask):
    """OpenICL Inference Task.

    This task is used to run the inference process.
    """

    name_prefix = "OpenICLInferMerged"
    log_subdir = "logs/infer"
    output_subdir = "predictions"

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)
        run_cfg = self.model_cfgs[0].get("run_cfg", {})
        self.num_gpus = run_cfg.get("num_gpus", 0)
        self.num_procs = run_cfg.get("num_procs", 1)
        self.nnodes = run_cfg.get('nnodes', 1)
        self.node_rank = run_cfg.get('node_rank', 0)
        self.master_addr = run_cfg.get('master_addr', "localhost")
        self.logger = get_logger()
        self.entry = []
        self.golds = []
        self.inferencer = None
        self.model_cfg = None
        self.out_path = ""
        self.ice_template = None
        self.prompt_template = None

    def get_command(self, cfg_path, template):
        """Get the command template for the task.

        Args:
            cfg_path (str): The path to the config file of the task.
            template (str): The template which have '{task_cmd}' to format
                the command.
        """
        sys.path.append(os.getcwd())
        script_path = __file__
        backend_keys = ["VLLM", "Lmdeploy"]
        use_backend = any(
            key in str(self.model_cfgs[0].get("type", ""))
            or key in str(self.model_cfgs[0].get("llm", {}).get("type", ""))
            for key in backend_keys
        )
        if self.num_gpus > 1 and not use_backend and self.nnodes == 1:
            port = random.randint(12000, 32000)
            command = (
                f"torchrun --master_port={port} "
                f"--nproc_per_node {self.num_procs} "
                f"{script_path} {cfg_path}"
            )
        elif self.nnodes > 1:
            port = 12345
            command = (f'torchrun --master_port={port} '
                       f'--nproc_per_node {self.num_procs} '
                       f'--nnodes {self.nnodes} '
                       f'--node_rank {self.node_rank} '
                       f'--master_addr {self.master_addr} '
                       f'{script_path} {cfg_path}')
        else:
            python = sys.executable
            command = f"{python} {script_path} {cfg_path}"

        return template.format(task_cmd=command)

    def run(self, cur_model=None, cur_model_abbr=None):
        self.logger.info(f"Task {task_abbr_from_cfg(self.cfg)}")

        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            self.max_out_len = model_cfg.get("max_out_len", None)
            self.batch_size = model_cfg.get("batch_size", None)
            self.min_out_len = model_cfg.get("min_out_len", None)
            if cur_model and cur_model_abbr == model_abbr_from_cfg(model_cfg):
                self.model = cur_model
            else:
                self.model = build_model_from_cfg(model_cfg)

            num_return_sequences = getattr(model_cfg, 'generation_kwargs', {}).pop('num_return_sequences', 1)
            _check_type(num_return_sequences, int)
            assert num_return_sequences > 0, f"num_return_sequences expected a positive integer, but got {num_return_sequences}"

            for dataset_cfg in dataset_cfgs:
                self.model_cfg = model_cfg
                self.dataset_cfg = dataset_cfg

                if 'n' not in self.dataset_cfg:
                    self.dataset_cfg['n'] = num_return_sequences
                _check_type(self.dataset_cfg['n'], int)
                assert self.dataset_cfg['n'] > 0, f"n expected a positive integer, but got {self.dataset_cfg['n']}"
                
                self.infer_cfg = self.dataset_cfg["infer_cfg"]
                self.dataset = build_dataset_from_cfg(self.dataset_cfg)
                self.build_inference()
                self.sub_cfg = {
                    "models": [self.model_cfg],
                    "datasets": [[self.dataset_cfg]],
                }
                entry, golds = self.get_data_list()
                self.entry.extend(entry)
                self.golds.extend(golds)
            self.do_inference()

    def build_inference(self):
        inferencer_cfg = self.infer_cfg["inferencer"]
        inferencer_cfg["model"] = self.model
        self._set_default_value(inferencer_cfg, "max_out_len", self.max_out_len)
        self._set_default_value(inferencer_cfg, "min_out_len", self.min_out_len)
        self._set_default_value(inferencer_cfg, "batch_size", self.batch_size)
        self._set_default_value(inferencer_cfg, "stopping_criteria", [])
        inferencer_cfg["max_seq_len"] = self.model_cfg.get("max_seq_len")
        self.inferencer = ICL_INFERENCERS.build(inferencer_cfg)

    def get_data_list(self):
        self.logger.info(f"Start load data of {task_abbr_from_cfg(self.sub_cfg)}")

        assert hasattr(self.infer_cfg, "ice_template") or hasattr(
            self.infer_cfg, "prompt_template"
        ), "Both ice_template and prompt_template cannot be None simultaneously."  # noqa: E501
        if hasattr(self.infer_cfg, "ice_template"):
            ice_template = ICL_PROMPT_TEMPLATES.build(self.infer_cfg["ice_template"])

        if hasattr(self.infer_cfg, "prompt_template"):
            prompt_template = ICL_PROMPT_TEMPLATES.build(
                self.infer_cfg["prompt_template"]
            )

        retriever_cfg = self.infer_cfg["retriever"].copy()
        retriever_cfg["dataset"] = self.dataset
        retriever = ICL_RETRIEVERS.build(retriever_cfg)

        # set inferencer's default value according to model's config'

        if hasattr(self.infer_cfg, "prompt_template") and hasattr(
            self.infer_cfg, "ice_template"
        ):
            return self.inferencer.get_data_list(
                retriever, ice_template=ice_template, prompt_template=prompt_template
            )
        elif hasattr(self.infer_cfg, "prompt_template"):
            return self.inferencer.get_data_list(
                retriever, prompt_template=prompt_template
            )
        else:
            return self.inferencer.get_data_list(retriever, ice_template=ice_template)

    def do_inference(self):
        out_path = get_infer_merged_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        self.inferencer.update_model_cfg(self.model_cfg)
        out_dir, out_name = osp.split(out_path)
        mkdir_or_exist(out_dir)
        self.inferencer.inference(self.entry, self.golds, out_dir, out_name)

    def get_log_path(self, file_extension: str = "json") -> str:
        """Get the path to the log file.

        Args:
            file_extension (str): The file extension of the log file.
                Default: 'json'.
        """
        return (
            get_perf_output_path(
                self.model_cfgs[0],
                self.dataset_cfgs[0][0],
                os.path.join(self.work_dir, self.log_subdir),
            )
            + "."
            + file_extension
        )

    def _set_default_value(self, cfg: ConfigDict, key: str, value: Any):
        if key not in cfg:
            cfg[key] = value


def parse_args():
    parser = argparse.ArgumentParser(description="Model Inferencer")
    parser.add_argument("config", help="Config file path")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.perf_counter()
    inferencer = OpenICLInferMergedTask(cfg)
    inferencer.run()
    end_time = time.perf_counter()
    get_logger().info(f"time elapsed: {end_time - start_time:.2f}s")
