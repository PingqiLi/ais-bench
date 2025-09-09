import inspect
import json
import h5py
import os
import os.path as osp
import time
import multiprocessing
from typing import List, Optional, Tuple

import mmengine
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from mmengine.config import ConfigDict

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.utils import batched
from ais_bench.benchmark.utils.build import build_perf_metric_calculator_from_cfg
from ais_bench.benchmark.utils.types import convert_positive_integers, check_output_config_from_meta_json

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import GenInferencerOutputHandler
from ais_bench.benchmark.utils.results import dump_results_dict, fast_dump_results_dict, dump_list_as_h5
from .icl_gen_inferencer import GenInferencer

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class GenPerfInferencer(GenInferencer):
    def __init__(
        self,
        model: BaseModel,
        max_out_len: int,
        stopping_criteria: Optional[List[str]] = None,
        max_seq_len: Optional[int] = None,
        min_out_len: Optional[int] = None,
        batch_size: Optional[int] = 1,
        gen_field_replace_token: Optional[str] = "",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        output_json_filename: Optional[str] = "performances",
        is_synthetic: Optional[bool] = False,
        num_prompts: int = None,
        **kwargs,
    ):
        super().__init__(
            model,
            max_out_len,
            stopping_criteria,
            max_seq_len,
            min_out_len,
            batch_size,
            gen_field_replace_token,
            output_json_filepath,
            output_json_filename,
            1,
            is_synthetic,
            **kwargs,
        )
        self.metrics_calculator = None
        self.num_prompts = num_prompts

    def get_data_list(
        self,
        retriever: BaseRetriever,
        ice_template: Optional[PromptTemplate] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> Tuple[List, List]:
        """
        Retrieves and processes data for inference.
        """
        ice_idx_list = retriever.retrieve()
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template,
        )
        ds_reader = retriever.dataset_reader
        if ds_reader.max_tokens_column:
            self.max_out_lens:List[int] = convert_positive_integers(ds_reader.dataset['test'][ds_reader.max_tokens_column],
                                                                    ds_reader.max_tokens_column)
        elif check_output_config_from_meta_json(self.meta_json_conf):
            self.max_out_lens = self.get_max_token_list_from_meta_json_file(self.meta_json_conf["output_config"],
                                                                            len(prompt_list))
        else:
            logger.info("Use model defined 'max_out_len' to control model max_out_tokens.")
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset["test"][ds_reader.output_column]
            prompt_list = list(zip(prompt_list, gold_ans))

        entry, golds = self.extract_data(ds_reader, prompt_list)
        return entry, golds

    def inference(
        self,
        entry: List[str],
        golds: List[Optional[str]],
        output_filepath: Optional[str] = None,
        output_filename: Optional[str] = None,
    ) -> List[str]:
        """
        Runs inference on the given entries and logs performance metrics.
        """
        if self.num_prompts:
            logger.info(f"The number of prompts to evaluate is {self.num_prompts}")
            entry = entry[:self.num_prompts]
        if self.is_synthetic:
            self.model.set_synthetic()

        output_handler = GenInferencerOutputHandler()
        output_filepath = output_filepath or self.output_json_filepath
        output_filename = output_filename or self.output_json_filename
        self.rps_plot_path = os.path.join(output_filepath, output_filename)

        logger.info("Starting performance inference process...")
        start_time_stamp = time.perf_counter()

        # Prepare inference parameters
        extra_gen_kwargs = self._build_extra_gen_kwargs()
        # Run inference
        with torch.no_grad():
            parsed_entries = self.model.parse_template(entry, mode='gen')
            results = self.inference_with_multi_process(
                self.model, self.model_cfg, parsed_entries, golds, **extra_gen_kwargs)
        logger.info("Start extracting pref datas ...")
        preds = self.extract_preds(results)
        logger.info("Finish extracting pref datas!")
        task_params = {"max_concurrency": self.batch_size}

        end_time_stamp = time.perf_counter()

        if self.is_main_process:
            os.makedirs(output_filepath, exist_ok=True)
            perf_details = {
                "task": task_params,
                "requests": preds,
            }
            logger.info("Dumping detail perf data ...")
            dump_start = time.perf_counter()
            dump_list_as_h5(
                perf_details["requests"]["decode_token_latencies"],
                osp.join(output_filepath, output_filename + "_details.h5")
            )
            del perf_details["requests"]["decode_token_latencies"]
            fast_dump_results_dict(
                perf_details,
                osp.join(output_filepath, output_filename + "_details.json")
            )
            logger.info(f"Dump detail perf data cost: {time.perf_counter() - dump_start}(s)")

        if self.dump_timer and self.is_main_process:
            timer_filepath = osp.join(output_filepath, "timer", "time.jsonl")
            os.makedirs(os.path.dirname(timer_filepath), exist_ok=True)
            time_dict = {
                "dataset_name": output_filename,
                "time": end_time_stamp - start_time_stamp,
                "num_sample": len(entry),
            }
            with open(timer_filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(time_dict) + "\n")

        logger.info(f"Performance task finished, results saved in {output_filepath}")
        return [sample["prediction"] for sample in output_handler.results_dict.values()]

    def extract_preds(self, results: List[dict]) -> dict:
        """
        Extracts predictions from generated results.
        """
        keys = list(results[0].keys())
        preds = {
            k: [
                pred.get(k)
                for pred in results
            ]
            for k in keys
        }
        preds["is_success"] = [pred.get("is_success", False) for pred in results]
        preds["is_empty"] = [pred.get("is_empty", False) for pred in results]
        del preds["input_data"]
        del preds["output"]
        return preds


