"""Direct Generation Inferencer."""

import inspect
import json
import os
import os.path as osp
import time
import multiprocessing
from pathlib import Path
from multiprocessing import RLock, freeze_support
from typing import List, Optional, Tuple, Any

import torch
import shutil
from tqdm import tqdm

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.utils import batched, build_model_from_cfg
from ais_bench.benchmark.global_consts import WORKERS_NUM


from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler

logger = get_logger(__name__)

DEFAULT_MAX_CONCURRENCY_PER_PROCESS = 500

def submit_single_model(model_cfg, mp_queue, **extra_gen_kwargs):
    model = build_model_from_cfg(model_cfg)
    if extra_gen_kwargs.get("is_synthetic"):
        model.set_synthetic()
    if extra_gen_kwargs.get("do_performance"):
        model.set_performance()
    model.generate_from_queue(
        mp_queue,
        **extra_gen_kwargs,
    )
    if not hasattr(model, "set_performance"):
        raise AttributeError(f'{model} has no except outputs, please check model config')
    return model.get_performance_data()

    
@ICL_INFERENCERS.register_module()
class GenInferencer(BaseInferencer):
    """Generation Inferencer class to directly evaluate by generation.

    Attributes:
        model (:obj:`BaseModelWrapper`, optional): The module to inference.
        max_seq_len (:obj:`int`, optional): Maximum number of tokenized words
            allowed by the LM.
        min_out_len (:obj:`int`, optional): Minimum number of generated tokens
            by the LM
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters. Defaults to 1.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the
            :obj:`model.generate()` method.
    """

    def __init__(
            self,
            model: BaseModel,
            max_out_len: int,
            stopping_criteria: List[str] = [],
            max_seq_len: Optional[int] = None,
            min_out_len: Optional[int] = None,
            batch_size: Optional[int] = 1,
            gen_field_replace_token: Optional[str] = '',
            output_json_filepath: Optional[str] = './icl_inference_output',
            output_json_filename: Optional[str] = 'predictions',
            save_every: Optional[int] = 1,
            is_synthetic: Optional[bool] = False,
            **kwargs) -> None:
        super().__init__(
            model=model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )

        self.gen_field_replace_token = gen_field_replace_token
        self.max_out_len = max_out_len
        self.min_out_len = min_out_len
        self.stopping_criteria = stopping_criteria
        self.dump_timer = kwargs.get('dump_timer', False)
        self.disable_cb = kwargs.get("disable_cb", False)

        if self.model.is_api and save_every is None:
            save_every = 1
        self.save_every = save_every
        self.is_synthetic = is_synthetic
        self.tmp_result_ids = []

    def inference_with_multi_process(
        self, model, model_cfg, inputs, golds, **extra_gen_kwargs
    ):
        if hasattr(model, "sync_rank") and model.sync_rank:
            inputs = model.sync_inputs(inputs)
        results = []
        if len(inputs) <= 0:
            logger.warning(f"Inputs data number is {len(inputs)}, result will be empty")
            return results
        max_concurrency = extra_gen_kwargs.get("batch_size", 1)
        
        # Maximum MAX_CONCURRENCY_PER_PROCESS concurrency per process, number of processes less than number of cores
        workers_num = min(
            multiprocessing.cpu_count(), (max_concurrency - 1) // DEFAULT_MAX_CONCURRENCY_PER_PROCESS + 1
        )
        if isinstance(WORKERS_NUM, int):
            if WORKERS_NUM > 0:
                logger.info(f"Get WORKERS_NUM :{WORKERS_NUM}")
                workers_num = min(WORKERS_NUM, multiprocessing.cpu_count())
        else:
            logger.warning(f"Expected WORKERS_NUM type int, but got {type(WORKERS_NUM)}. Has been reset to {workers_num}")
        logger.info(f"Concurrency is set to {max_concurrency}, infer with total {workers_num} process")
        q, r = divmod(max_concurrency, workers_num)
        concurrencys = [q + 1] * r + [q] * (workers_num - r)
        task_data_num = len(inputs) - len(self.tmp_result_ids)
        if task_data_num != len(inputs):
            logger.info(f"{len(self.tmp_result_ids)} requests have been completed, requests remaining: {task_data_num}")
        q, r = divmod(len(inputs), workers_num)
        data_bucket_sizes = [q + 1] * r + [q] * (workers_num - r)
        with multiprocessing.Manager() as manager:
            data_buckets = []
            real_data_nums = []
            bucket_index = 0
            data_index = 0 
            while data_index <len(inputs):
                bucket_size = data_bucket_sizes[bucket_index]
                mp_queue = manager.Queue(bucket_size + 1)
                real_data_num = 0
                while bucket_size > 0:
                    if data_index not in self.tmp_result_ids:
                        try:
                            mp_queue.put(dict(data_id=data_index, prompt=inputs[data_index], gold=golds[data_index]))
                            real_data_num += 1
                        except IndexError as e:
                            logger.error(f"data index out of range")
                            return results
                    data_index += 1
                    bucket_size -= 1
                bucket_index += 1
                mp_queue.put(None)
                data_buckets.append(mp_queue)
                real_data_nums.append(real_data_num)
        
            request_rate = model.request_rate
            if request_rate < 0.1:
                logger.info(f"get request_rate {request_rate} small than 0.1, all requests will send together!")
                request_rate = 0
            request_rate_mean = request_rate / workers_num
            max_data_bucket_size = max(data_bucket_sizes)
            # Set the timing of token release according to qps, only one request can hold the token at each moment
            freeze_support()
            pool = multiprocessing.Pool(processes=workers_num, initializer=tqdm.set_lock, initargs=(RLock(),))
            async_results = []
            for i in range(workers_num):
                new_gen_kwargs = extra_gen_kwargs.copy()
                new_gen_kwargs.update({
                    "concurrency": max(1, concurrencys[i]),
                    "ori_nums":    data_bucket_sizes[i],
                    "data_nums":   real_data_nums[i],
                    "process_id":  i,
                    "qps":         request_rate_mean * data_bucket_sizes[i] / max_data_bucket_size
                })
                res = pool.apply_async(func=submit_single_model, 
                                        args=(model_cfg, data_buckets[i],),
                                        kwds=new_gen_kwargs,
                                        error_callback=lambda x:logger.error(x)
                                        )
                async_results.append(res)
            pool.close()
            pool.join()
            for res in async_results:
                results.extend(res.get())
        return results

    def extract_data(self, ds_reader, datum: Any) -> Tuple[List, List]:
        """
        Extracts input entries and corresponding gold answers from the given datum using the dataset reader.

        Args:
            ds_reader: The dataset reader object.
            datum: Data sample from the dataloader.

        Returns:
            A tuple of two lists: (entries, gold answers).
        """
        if ds_reader.output_column:
            if self.batch_size is not None:
                entry, golds = list(zip(*datum))
            else:
                entry = [datum[0]]
                golds = [datum[1]]
        else:
            entry = datum
            golds = [None for _ in range(len(entry))]
        return entry, golds

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
        extra_kwargs['is_synthetic'] = self.is_synthetic
        extra_kwargs['batch_size'] = self.batch_size
        extra_kwargs['max_out_len'] = self.max_out_len
        if  hasattr(self.model, "set_performance"):
            extra_kwargs['do_performance'] = self.model.do_performance
        return extra_kwargs
                     
    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 0. Set synthetic if needed
        if self.is_synthetic:
            self.model.set_synthetic()

        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 3.1 Fetch and zip prompt & gold answer if output column exists
        ds_reader = retriever.dataset_reader
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset['test'][ds_reader.output_column]
            prompt_list = list(zip(prompt_list, gold_ans))

        extra_gen_kwargs = self._build_extra_gen_kwargs()
        num_return_sequences = getattr(self.model, 'generation_kwargs', {}).get('num_return_sequences', 1)
        all_success = True
        if not self.disable_cb :
            tmp_json_filepath = os.path.join(output_json_filepath,
                'tmp_' + output_json_filename.split('.')[0])
            output_handler.load_tmp_result(tmp_json_filepath)
            for data_id in output_handler.results_dict.keys():
                self.tmp_result_ids.append(int(data_id))
            extra_gen_kwargs.update({"tmp_result_dir": tmp_json_filepath})
            start_time_stamp = time.perf_counter()
            entry, golds = self.extract_data(ds_reader, prompt_list)
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entry, mode='gen')
                results = self.inference_with_multi_process(
                    self.model, self.model_cfg, parsed_entries, golds, **extra_gen_kwargs)
                results.sort(key=lambda x: x['id'])
                generated = [result['output'] for result in results]
            for prediction in batched(results, num_return_sequences):
                if num_return_sequences == 1:
                    prediction = prediction[0]
                if not prediction.get('is_success'):
                    all_success = False
                    pred = ""
                else:
                    pred = prediction.get('output')
                data_id = prediction.get('id')
                if data_id >= len(golds) or data_id < 0:
                    raise IndexError(f"No gold of output id {data_id}")
                output_handler.save_results(parsed_entries[data_id],
                                            pred,
                                            data_id,
                                            gold=golds[data_id])
        else:
            # Create tmp json file for saving intermediate results and future
            # resuming
            tmp_json_filepath = os.path.join(output_json_filepath,
                            'tmp_' + output_json_filename)
            output_handler.load_tmp_result(tmp_json_filepath)
            index = len(output_handler.results_dict)

            # 4. Wrap prompts with Dataloader
            logger.info('Starting build dataloader')
            dataloader = self.get_dataloader(prompt_list[index:], self.batch_size)

            # 5. Inference for prompts in each batch
            logger.info('Starting inference process...')

            start_time_stamp = time.perf_counter()
            num_sample = 0
            for datum in tqdm(dataloader, disable=not self.is_main_process):
                entry, golds = self.extract_data(ds_reader, datum)
                # 5-1. Inference with local model
                with torch.no_grad():
                    parsed_entries = self.model.parse_template(entry, mode='gen')
                    results = self.model.generate_from_template(
                        entry, **extra_gen_kwargs)
                    generated = results

                # 5-3. Save current output
                for prompt, prediction, gold in zip(
                        parsed_entries, batched(generated, num_return_sequences),
                        golds):
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
                num_sample += len(datum)

        end_time_stamp = time.perf_counter()

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
            if osp.exists(tmp_json_filepath):
                if osp.isdir(tmp_json_filepath) and all_success:
                    shutil.rmtree(tmp_json_filepath)
                elif osp.isfile(tmp_json_filepath):
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

    def get_generation_prompt_list_from_retriever_indices(
            self,
            ice_idx_list: List[List[int]],
            retriever: BaseRetriever,
            gen_field_replace_token: str,
            max_seq_len: Optional[int] = None,
            ice_template: Optional[PromptTemplate] = None,
            prompt_template: Optional[PromptTemplate] = None):
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=gen_field_replace_token,
                ice_template=ice_template,
                prompt_template=prompt_template)
            if max_seq_len is not None:
                prompt_token_num = self.model.get_token_len_from_template(
                    prompt, mode='gen')
                while len(ice_idx) > 0 and prompt_token_num > max_seq_len:
                    ice_idx = ice_idx[:-1]
                    ice = retriever.generate_ice(ice_idx,
                                                 ice_template=ice_template)
                    prompt = retriever.generate_prompt_for_generate_task(
                        idx,
                        ice,
                        gen_field_replace_token=gen_field_replace_token,
                        ice_template=ice_template,
                        prompt_template=prompt_template)
                    prompt_token_num = self.model.get_token_len_from_template(
                        prompt, mode='gen')
            prompt_list.append(prompt)
        return prompt_list


@ICL_INFERENCERS.register_module()
class GLMChoiceInferencer(GenInferencer):

    def __init__(self, *args, choices=['A', 'B', 'C', 'D'], **kwargs):
        super().__init__(*args, **kwargs)
        self.choices = choices

    def inference(self,
                  retriever: BaseRetriever,
                  ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None,
                  output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template)

        # 4. Wrap prompts with Dataloader
        dataloader = self.get_dataloader(prompt_list, self.batch_size)
        index = 0

        # 5. Inference for prompts in each batch
        logger.info('Starting inference process...')
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            # 5-1. Inference with local model
            with torch.no_grad():
                parsed_entries = self.model.parse_template(entry, mode='gen')
                results = self.model.choice(entry, choices=self.choices)
                generated = results

            # 5-3. Save current output
            for prompt, prediction in zip(parsed_entries, generated):
                output_handler.save_results(prompt, prediction, index)
                index = index + 1

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath,
                                         output_json_filename)
        return [
            sample['prediction']
            for sample in output_handler.results_dict.values()
        ]
