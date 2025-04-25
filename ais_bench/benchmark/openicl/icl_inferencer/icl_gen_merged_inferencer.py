import inspect
import json
import os
import os.path as osp
import time
from typing import List, Optional, Tuple

import mmengine
import torch
from tqdm import tqdm
import math
from concurrent.futures import ThreadPoolExecutor

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.utils import batched
from ais_bench.benchmark.utils.results import MetricsCalculator

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import dump_results_dict, GenInferencerOutputHandler
from .icl_gen_inferencer import GenInferencer

logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class GenMergedInferencer(GenInferencer):
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
        output_json_filename: Optional[str] = "predictions",
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
        self.concurrency = kwargs.get('concurrency')
        self.is_synthetic = is_synthetic

    def _build_extra_gen_kwargs(self) -> dict:
        """
        Builds extra keyword arguments for the model's generate method based on its signature.

        Returns:
            A dictionary of extra keyword arguments.
        """
        extra_kwargs = {}
        sig = inspect.signature(self.model.generate)
        if 'stopping_criteria' in sig.parameters:
            extra_kwargs['stopping_criteria'] = self.stopping_criteria
        if 'min_out_len' in sig.parameters:
            extra_kwargs['min_out_len'] = self.min_out_len
        extra_kwargs['batch_size'] = self.batch_size
        return extra_kwargs

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
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> List[str]:
        """
        Runs inference on the given entries and logs performance metrics.
        """
        if self.is_synthetic:
            self.model.set_synthetic()

        output_handler = GenInferencerOutputHandler()
        tmp_json_filepath = os.path.join(output_json_filepath,
                                         'tmp_' + output_json_filename)

        if self.concurrency: # continous batch run
            if self.concurrency > 4096 or self.concurrency <= 0:
                logger.warning(f'concurrency must be in [1, 4096], but get {self.concurrency}, '
                               'continous infer will not be enable')
            else:
                logger.warning('The concurrency is set, continous infer will be turned on, '
                           'intermediate results will not be saved')
            extra_gen_kwargs = self._build_extra_gen_kwargs()
            num_return_sequences = getattr(self.model, 'generation_kwargs', {}).get('num_return_sequences', 1)
            start_time_stamp = time.perf_counter()
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entry, mode='gen')
                results = self.model.generate_from_template(
                    entry, max_out_len=self.max_out_len, **extra_gen_kwargs)
                generated = results
            index = 0
            for prompt, prediction, gold in zip(
                    parsed_entries, batched(generated, num_return_sequences),
                    golds):

                if num_return_sequences == 1:
                    prediction = prediction[0]
                output_handler.save_results(prompt,
                                            prediction,
                                            index,
                                            gold=gold)
                index += 1

        else: # static batch run
            if osp.exists(tmp_json_filepath):
                # TODO: move resume to output handler
                try:
                    tmp_result_dict = mmengine.load(tmp_json_filepath)
                except Exception:
                    pass
                else:
                    output_handler.results_dict = tmp_result_dict
                    index = len(tmp_result_dict)
            else:
                index = 0

            logger.info('Starting inference process...')

            start_time_stamp = time.perf_counter()
            num_sample = 0
            total_ds_len = len(entry)
            if total_ds_len != len(golds):
                raise ValueError("length of entry and golds is not equal")

            end_round = math.ceil(total_ds_len / self.batch_size)
            start_round = math.floor((index + 1) / self.batch_size)
            index = self.batch_size * start_round

            for i in tqdm(range(start_round, end_round), desc="Batch Infer Processing", unit="batch", dynamic_ncols=True):
                if total_ds_len % self.batch_size != 0 and i == end_round - 1:
                    entry_per_bs = entry[i * self.batch_size:]
                    golds_per_bs = golds[i * self.batch_size:]
                else:
                    entry_per_bs = entry[(i) * self.batch_size: (i + 1) * self.batch_size]
                    golds_per_bs = golds[(i) * self.batch_size: (i + 1) * self.batch_size]

                # 5-1. Inference with local model
                extra_gen_kwargs = {}
                sig = inspect.signature(self.model.generate)
                if 'stopping_criteria' in sig.parameters:
                    extra_gen_kwargs['stopping_criteria'] = self.stopping_criteria
                if 'min_out_len' in sig.parameters:
                    extra_gen_kwargs['min_out_len'] = self.min_out_len
                with torch.no_grad():
                    parsed_entries = self.model.parse_template(entry_per_bs, mode='gen')
                    results = self.model.generate_from_template(
                        entry_per_bs, max_out_len=self.max_out_len, **extra_gen_kwargs)
                    generated = results

                num_return_sequences = getattr(self.model, 'generation_kwargs',
                                            {}).get('num_return_sequences', 1)
                # 5-3. Save current output
                for prompt, prediction, gold in zip(
                        parsed_entries, batched(generated, num_return_sequences),
                        golds_per_bs):
                    if num_return_sequences == 1:
                        prediction = prediction[0]
                    output_handler.save_results(prompt,
                                                prediction,
                                                index,
                                                gold=gold)
                    index = index + 1

                # 5-4. Save intermediate results
                if (self.save_every is not None and index % self.save_every == 0
                        and self.is_main_process):
                    output_handler.write_to_json(output_json_filepath,
                                                'tmp_' + output_json_filename)
                num_sample += 1

        end_time_stamp = time.perf_counter()

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        if self.dump_timer and self.is_main_process:
            timer_filepath = os.path.join(output_json_filepath, 'timer',
                                          'time.jsonl')
            os.makedirs(os.path.dirname(timer_filepath), exist_ok=True)
            time_dict = {
                'dataset_name': output_json_filename.removesuffix('.json'),
                'time': end_time_stamp - start_time_stamp,
                'num_sample': num_sample
            }
            with open(timer_filepath, 'a') as f:
                f.write(json.dumps(time_dict) + '\n')

        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]

    def extract_preds(self, results: List[dict]) -> dict:
        """
        Extracts predictions from generated results.
        """
        keys = list(results[0].keys())
        preds = {
            k: [
                pred.get(k)
                for pred in results
                if pred.get("is_success") and not pred.get("is_empty")
            ]
            for k in keys
        }
        preds["is_success"] = [pred.get("is_success", False) for pred in results]
        preds["is_empty"] = [pred.get("is_empty", False) for pred in results]
        return preds
