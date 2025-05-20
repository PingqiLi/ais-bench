import inspect
import os
import os.path as osp
from typing import List, Optional, Tuple

import torch
from tqdm import tqdm

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.registry import ICL_INFERENCERS

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from ais_bench.benchmark.utils.results import dump_results_dict
from .icl_gen_inferencer import GenInferencer

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class GenModelPerfInferencer(GenInferencer):
    def __init__(
        self,
        model: BaseModel,
        max_out_len: int,
        stopping_criteria: Optional[List[str]] = [],
        max_seq_len: Optional[int] = None,
        min_out_len: Optional[int] = None,
        batch_size: Optional[int] = 1,
        gen_field_replace_token: Optional[str] = "",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        output_json_filename: Optional[str] = "performances",
        is_synthetic: Optional[bool] = False,
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
        self.stopping_criteria = stopping_criteria
        self.concurrency = batch_size if batch_size else 1

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
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset["test"][ds_reader.output_column]
            prompt_list = list(zip(prompt_list, gold_ans))
        entry = [p[0] for p in prompt_list] if ds_reader.output_column else prompt_list
        golds = (
            [p[1] for p in prompt_list]
            if ds_reader.output_column
            else [None] * len(entry)
        )
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
        if self.is_synthetic:
            self.model.set_synthetic()

        self.model.set_performance()

        output_filepath = output_filepath or self.output_json_filepath
        output_filename = output_filename or self.output_json_filename

        logger.info("Starting performance inference process...")

        # Prepare inference parameters
        extra_gen_kwargs = self._build_extra_gen_kwargs()

        # Run inference
        with torch.no_grad():
            assert isinstance(entry, list) and len(entry) >= self.concurrency
            for i in tqdm(range(0, len(entry), self.concurrency), desc="Processing", unit="batch", dynamic_ncols=True):
                data = entry[i:i + self.concurrency]
                results = self.model.generate_from_template(
                        data, **extra_gen_kwargs)
        perf_results = self.model.handle_perf_result(output_filepath, output_filename)

        #Save performance results
        if self.is_main_process:
            os.makedirs(output_filepath, exist_ok=True)
            dump_results_dict(
                perf_results,
                osp.join(output_filepath, output_filename + ".json"),
            )
        #Summary
        logger.info(f"Performance results is {perf_results}")
        logger.info(f"Performance task finished, results saved in {output_filepath}")
        return