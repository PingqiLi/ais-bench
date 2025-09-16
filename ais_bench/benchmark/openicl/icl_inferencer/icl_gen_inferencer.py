"""Direct Generation Inferencer."""

import inspect
import json
import os
import os.path as osp
import time
import math
import multiprocessing
from pathlib import Path
from multiprocessing import RLock, freeze_support
from typing import List, Optional, Tuple, Any, Literal, Dict
from mmengine.config import ConfigDict

import torch
import shutil
import numpy as np
from tqdm import tqdm
import itertools

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.utils import batched, build_model_from_cfg
from ais_bench.benchmark.global_consts import WORKERS_NUM
from ais_bench.benchmark.utils.types import convert_positive_integers, check_output_config_from_meta_json, safe_convert
from ais_bench.benchmark.utils.rps_distribution_plot import plot_rps_distribution, add_actual_rps_to_chart

from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
import concurrent.futures

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
    if model.interrupted:
        return model.get_performance_data_backup()
    return model.get_performance_data()


def _count_elements(nested_list: list) -> int:
    """Count the number of elements in arbitrarily nested lists."""
    count = 0
    stack = [iter(nested_list)]
    
    while stack:
        current = stack[-1]
        try:
            item = next(current)
            if isinstance(item, list):
                stack.append(iter(item))
            else:
                count += 1
        except StopIteration:
            stack.pop()
    
    return count


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
        self.num_return_sequences = getattr(self.model, 'generation_kwargs', {}).get('num_return_sequences', 1)

        self.dump_timer = kwargs.get('dump_timer', False)
        self.disable_cb = kwargs.get("disable_cb", False)
        self.meta_json_conf = kwargs.get("meta_json_conf", {})

        if self.model.is_api and save_every is None:
            save_every = 1
        self.save_every = save_every
        self.is_synthetic = is_synthetic
        self.tmp_result_ids = []
        self.max_out_lens = []
        self.functioncall_infos = []
        self.rps_plot_path = ""

    def inference_with_multi_process(
        self, model, model_cfg, inputs, golds, **extra_gen_kwargs
    ):
        if hasattr(model, "sync_rank") and model.sync_rank:
            inputs = model.sync_inputs(inputs)
        results = []
        total_requests_num = len(inputs)
        total_requests_size = _count_elements(inputs)

        if total_requests_num <= 0:
            logger.warning(f"Inputs data number is {total_requests_num}, result will be empty")
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
        if workers_num > total_requests_num:
            logger.warning(f"Number of processes {workers_num} is greater than the number of inputs {total_requests_num}, has been reset to {total_requests_num}")
            workers_num = total_requests_num
        logger.info(f"Concurrency is set to {max_concurrency}, infer with total {workers_num} process")
        q, r = divmod(max_concurrency, workers_num)
        concurrencys = [q + 1] * r + [q] * (workers_num - r)
        task_data_num = total_requests_num - len(self.tmp_result_ids)
        if task_data_num != total_requests_num:
            logger.info(f"{len(self.tmp_result_ids)} requests have been completed, requests remaining: {task_data_num}")
        q, r = divmod(total_requests_num, workers_num)
        data_bucket_sizes = [q + 1] * r + [q] * (workers_num - r)
        with multiprocessing.Manager() as manager:
            data_buckets = []
            real_data_nums = []
            bucket_index = 0
            data_index = 0
            while data_index < total_requests_num:
                bucket_size = data_bucket_sizes[bucket_index]
                mp_queue = manager.Queue(bucket_size + 1)
                real_data_num = 0
                while bucket_size > 0:
                    if data_index not in self.tmp_result_ids:
                        try:
                            data_dict = dict(
                                data_id=data_index,
                                prompt=inputs[data_index],
                                gold=golds[data_index],
                                max_tokens=(
                                    self.max_out_len
                                    if data_index >= len(self.max_out_lens)
                                    else self.max_out_lens[data_index]
                                ),
                            )
                            if data_index < len(self.functioncall_infos):
                                data_dict.update(self.functioncall_infos[data_index])
                            mp_queue.put(data_dict)
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
            
            global_offsets_dict = self.get_global_offsets_dict(
                total_requests_num=total_requests_size,
                workers_num=workers_num,
                model=model,
                model_cfg=model_cfg,
            )

            max_data_bucket_size = max(data_bucket_sizes)
            # Set the timing of token release according to qps, only one request can hold the token at each moment
            freeze_support()
            pool = multiprocessing.Pool(processes=workers_num, initializer=tqdm.set_lock, initargs=(RLock(),))
            async_results = []

            global_start_time = time.perf_counter()
            for i in range(workers_num):
                new_gen_kwargs = extra_gen_kwargs.copy()
                new_gen_kwargs.update({
                    "concurrency": max(1, concurrencys[i]),
                    "ori_nums":    data_bucket_sizes[i],
                    "data_nums":   real_data_nums[i],
                    "process_id":  i,
                    "sleep_offsets": global_offsets_dict[i],
                    "global_start_time": global_start_time,
                })
                res = pool.apply_async(func=submit_single_model,
                                        args=(model_cfg, data_buckets[i],),
                                        kwds=new_gen_kwargs,
                                        error_callback=lambda x:logger.error(x)
                                        )
                async_results.append(res)
            pool.close()
            try:
                pool.join()
            except KeyboardInterrupt:
                logger.warning(f"Request posting interrupted by User!")
                logger.warning(f"Detect interruption, waiting for subprocess finish current task ...")
                # Check whether process is finished
                start_time = time.time()
                timeout = 300  # 5 min
                while True:
                    all_done = all(res.ready() for res in async_results)
                    if all_done:
                        break
                    if time.time() - start_time > timeout:
                        logger.warning(f"Timeout {timeout}s, force terminate remaining process")
                        pool.terminate()
                        break
                    time.sleep(1)
                pool.join()  # Insure that resource is released

            iterables = [res for res in async_results]
            results = list(itertools.chain.from_iterable(res.get() for res in iterables))
            if getattr(model, "do_performance", False):
                post_time_list: list = [(each.get("start_time") - global_start_time) for each in results]
                add_actual_rps_to_chart(self.rps_plot_path, post_time_list)
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
        if hasattr(self.model, "set_performance"):
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
        self.rps_plot_path = os.path.join(output_json_filepath, output_json_filename)

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
        if ds_reader.max_tokens_column:
            self.max_out_lens:List[int] = convert_positive_integers(ds_reader.dataset['test'][ds_reader.max_tokens_column],
                                                                    ds_reader.max_tokens_column)
        elif check_output_config_from_meta_json(self.meta_json_conf):
            self.max_out_lens = self.get_max_token_list_from_meta_json_file(self.meta_json_conf["output_config"],
                                                                            len(prompt_list))
        else:
            logger.info("Use model defined 'max_out_len' to control model max_out_tokens.")
        extra_gen_kwargs = self._build_extra_gen_kwargs()
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
            if len(generated) != len(golds):
                all_success = False
            for predictions in batched(results, self.num_return_sequences):
                for prediction in predictions:
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
                for prompt, predictions, gold in zip(
                        parsed_entries, batched(generated, self.num_return_sequences),
                        golds):
                    for prediction in predictions:
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
        ds_reader = retriever.dataset_reader
        
        keys_of_interest = [
            "function",
            "id",
            "involved_classes",
            "initial_config",
            "missed_function",
        ]
        
        for i in range(len(ds_reader['test'])):
            row = ds_reader['test'][i]  
            entry = {}
            for k in keys_of_interest:
                if k in row:
                    entry[k] = row[k]
                else:
                    continue
                if k == 'id':
                    entry['data_name'] = entry.pop('id')
            self.functioncall_infos.append(entry)
        return prompt_list

    def get_max_token_list_from_meta_json_file(self, output_config: dict, prompt_length):
        method = output_config["method"]
        params = output_config["params"]
        logger.info("Distribution Summary: ")
        if method == "uniform":
            logger.info(f"--uniform distribution with min_value: {params['min_value']}, max_value: {params['max_value']}")
            max_token_list = np.random.randint(int(params["min_value"]), int(params["max_value"]) + 1, prompt_length)
            return [int(token) for token in max_token_list]
        elif method == "percentage":
            max_token_list = []
            show_log_info = []
            for max_tokens, rate in params["percentage_distribute"]:
                max_token_list.extend([max_tokens] * math.floor(rate * prompt_length))
                show_log_info.append([max_tokens, rate*100, math.floor(rate * prompt_length)])
            # TODO Fix the situation where the product is not integer
            if len(max_token_list) < prompt_length:
                max_token_list.extend([params["percentage_distribute"][-1][0]] * (prompt_length - len(max_token_list)))
                show_log_info[-1][2] += prompt_length - len(max_token_list)
            for out_token_len, rate, request_num in show_log_info:
                logger.info("--max_out_token: {},  ratio: {:.1f}%,  request_num: {}".format(out_token_len, rate, request_num))
            return max_token_list
        else:
            raise ValueError(f"Unsupport data distribution types: {method}")


    def get_global_offsets_dict(
        self,
        total_requests_num: int,
        workers_num: int,
        model: BaseModel,
        model_cfg: ConfigDict,
        ) -> Dict[int, np.ndarray]:
        """
        Generate a global sleep offset dictionary for multi-process environments.
        
        Computes request timing offsets and distributes them across worker processes
        to coordinate request scheduling in distributed load testing scenarios.
        
        Args:
            total_requests_num: Total number of requests to generate
            workers_num: Number of worker processes
            model: Determine if do_performance
            model_cfg: Model configuration (contains traffic control parameters)
            
        Returns:
            Dictionary mapping process ID to numpy array of time offsets (seconds)
        """
        start_time = time.perf_counter()
        # Generate global timing offsets using vectorized computation
        global_offsets_arr = self._get_sleep_interval_offset_list(
            total_requests_num=total_requests_num,
            model=model,
            model_cfg=model_cfg,
        )
        
        # Distribute offsets across worker processes
        global_offsets_dict = {}
        for process_id in range(workers_num):
            global_offsets_dict[process_id] = self._get_process_sleep_offsets(
                global_offsets_arr,
                process_id,
                workers_num
            ).tolist()
        
        logger.info(f"Calculate global interval offsets time: {(time.perf_counter() - start_time):.4f} s")
        return global_offsets_dict


    def _get_sleep_interval_offset_list(
        self,
        total_requests_num: int,
        model: BaseModel,
        model_cfg: ConfigDict
        ) -> np.ndarray:
        """
        Generate global sleep time offsets (seconds) for request scheduling.
        
        Implements traffic control features:
        - Burstiness modeling using Gamma distribution
        - Ramp-up strategies (linear/exponential)
        - Uniform distribution baseline
        
        Args:
            total_requests_num: Total number of requests
            model: Determine if do_performance
            model_cfg: Configuration dictionary with traffic parameters:
                structure
                request_rate: float,  # Base request rate (RPS), default 0
                traffic_cfg =dict(
                    "burstiness": Optionl[float],    # Burst factor >= 0, default 0
                    "ramp_up_strategy": Optional[str],  # "linear"/"exponential"
                    "ramp_up_start_rps": Optional[float],  # Starting RPS
                    "ramp_up_end_rps": Optional[float]     # Ending RPS
                )
            
        Returns:
            Numpy array of cumulative time offsets (seconds)
        
        Notice:
            If ramp-up params are valid, ramp_up_end_rps will cover request_rate as the final rps.
        """
        # Constants
        FINAL_RPS_MINIMUM_THRESHOLD = 0.1 # minimum acceptable RPS
        MIN_RELIABLE_INTERVAL = 0.001 # minimum reliable time interval (1 millisecond)

        # Extract base request rate from configuration
        request_rate = model_cfg.get('request_rate', 0.0)
        
        # Initialize empty array for cumulative delays
        cumulative_delays = np.array([])
        
        # Extract traffic configuration parameters
        traffic_cfg = getattr(model_cfg, "traffic_cfg", {})

        burstiness = safe_convert(traffic_cfg.get("burstiness"), float, 0.0, param_name="burstiness")
        ramp_up_strategy = safe_convert(traffic_cfg.get("ramp_up_strategy"), str, None, param_name="ramp_up_strategy")
        ramp_up_start_rps = safe_convert(traffic_cfg.get("ramp_up_start_rps"), float, None, param_name="ramp_up_start_rps")
        ramp_up_end_rps = safe_convert(traffic_cfg.get("ramp_up_end_rps"), float, None, param_name="ramp_up_end_rps")
        
        # Validate ramp-up strategy parameters
        valid_strategies = ("linear", "exponential")
        if ramp_up_strategy not in valid_strategies:
            if ramp_up_strategy:
                logger.warning(
                    f"Invalid ramp_up_strategy: '{ramp_up_strategy}'. "
                    f"Valid options are {valid_strategies}. "
                    "Disabling ramp-up strategy."
                )
            ramp_up_strategy = None
        if ramp_up_strategy and (ramp_up_start_rps is None or ramp_up_end_rps is None):
            logger.warning(
                f"Ramp-up strategy '{ramp_up_strategy}' requires both "
                "ramp_up_start_rps and ramp_up_end_rps parameters. "
                "Disabling ramp-up strategy due to missing parameters."
            )
            ramp_up_strategy = None
        if ramp_up_strategy and (ramp_up_start_rps > ramp_up_end_rps):
            logger.warning(
                f"Invalid ramp-up parameters: ramp_up_start_rps ({ramp_up_start_rps}) "
                f"is greater than ramp_up_end_rps ({ramp_up_end_rps}). "
                "Ramp-up should start from lower to higher RPS. "
                "Disabling ramp-up strategy."
            )
            ramp_up_strategy = None

        if ramp_up_strategy is None:
            logger.info("The ramp-up strategy will not be adopted.")
        
        # Handle extremely low request rates (all requests sent simultaneously)
        rate_to_check = ramp_up_end_rps if ramp_up_strategy is not None else request_rate
        if rate_to_check < FINAL_RPS_MINIMUM_THRESHOLD:
            info_msg = f"Request rate ({request_rate})" if ramp_up_strategy is None else f"Ramp-up end rps ({ramp_up_end_rps})" + f" < {FINAL_RPS_MINIMUM_THRESHOLD}"
            logger.info(f"{info_msg}, sending all requests simultaneously")
            return np.array([])

        try:
            # Vectorized request rate calculation
            request_indices = np.arange(total_requests_num)
            progress = request_indices / max(total_requests_num - 1, 1)
            
            if ramp_up_strategy == "linear":
                request_rates = ramp_up_start_rps + (ramp_up_end_rps - ramp_up_start_rps) * progress
            elif ramp_up_strategy == "exponential":
                ratio = ramp_up_end_rps / ramp_up_start_rps
                request_rates = ramp_up_start_rps * (ratio ** progress)
            else:
                request_rates = np.full(total_requests_num, request_rate)
            
            # Handle invalid request rates
            request_rates = np.where(request_rates <= 0, 0, request_rates)
            
            # Generate inter-arrival times
            delays = np.zeros(total_requests_num)
            non_zero_mask = request_rates > 0
            
            if burstiness == 0:
                # # Use fixed intervals (no randomness)
                delays[non_zero_mask] = 1.0 / request_rates[non_zero_mask]
            else:
                # Generate inter-arrival times using Gamma distribution
                # Gamma(shape=k, scale=θ) where θ = 1/(λ·k)
                theta = 1.0 / (request_rates[non_zero_mask] * burstiness)
                delays[non_zero_mask] = np.random.gamma(
                    shape=burstiness, 
                    scale=theta, 
                    size=np.sum(non_zero_mask)
                )
            
            # Compute cumulative delays
            cumulative_delays = np.cumsum(delays)
            
            # Normalization for uniform distribution cases
            if ramp_up_strategy is None and cumulative_delays.size > 0 and cumulative_delays[-1] > 0:
                target_total = total_requests_num / request_rate
                normalize_factor = target_total / cumulative_delays[-1]
                cumulative_delays *= normalize_factor
                
            # Detect timing anomalies (intervals below minimum reliable threshold)
            timing_anomaly_mask = delays < MIN_RELIABLE_INTERVAL
            timing_anomaly_indices = np.where(timing_anomaly_mask)[0]
            
            # Calculate expected intervals (without burstiness effect)
            expected_intervals = np.zeros(total_requests_num)
            expected_intervals[non_zero_mask] = 1.0 / request_rates[non_zero_mask]
            
            # Calculate deviation ratio: |actual - expected| / expected
            interval_deviations = np.zeros(total_requests_num)
            interval_deviations[non_zero_mask] = np.abs(delays[non_zero_mask] - expected_intervals[non_zero_mask]) / expected_intervals[non_zero_mask]
            
            # Identify points significantly affected by burstiness (deviation > 50%)
            burstiness_anomaly_mask = interval_deviations > 0.5
            burstiness_anomaly_indices = np.where(burstiness_anomaly_mask)[0]
            
            # Remove duplicates (timing anomalies take precedence)
            if timing_anomaly_indices.size > 0 and burstiness_anomaly_indices.size > 0:
                timing_set = set(timing_anomaly_indices.tolist())
                burstiness_set = set(burstiness_anomaly_indices.tolist())

                intersection = timing_set & burstiness_set
                burstiness_set = burstiness_set - intersection
                burstiness_anomaly_indices = np.array(sorted(list(burstiness_set)), dtype=np.int64)

            # If burstiness=0, there should be no burstiness anomalies
            if burstiness == 0:
                burstiness_anomaly_indices = np.array([], dtype=np.int64)
            
            # Visualization for debugging purposes
            if cumulative_delays.size > 0 and getattr(model, "do_performance", False):
                logger.info("Begin to draw RPS distribution plot...")
                rps_plot_path = self.rps_plot_path if self.rps_plot_path else os.getcwd()
                rps_plot_path, _ = os.path.splitext(rps_plot_path)
                self.rps_plot_path = str(rps_plot_path) + "_rps_distribution_plot.html"
                try:
                    plot_rps_distribution(
                        cumulative_delays=cumulative_delays,
                        timing_anomaly_indices=timing_anomaly_indices,
                        burstiness_anomaly_indices=burstiness_anomaly_indices,
                        request_rate=request_rate,
                        burstiness=burstiness,
                        ramp_up_strategy=ramp_up_strategy,
                        ramp_up_start_rps=ramp_up_start_rps,
                        ramp_up_end_rps=ramp_up_end_rps,
                        output_path=self.rps_plot_path,
                    )
                except Exception as e:
                    logger.error(f"Error drawing RPS distribution plot: {e}")
        except Exception as e:
            logger.error(f"Error generating sleep offsets: {e}")
            logger.warning("Sending all requests simultaneously")
            return np.array([])
        
        return cumulative_delays


    def _get_process_sleep_offsets(
        self,
        global_offsets: np.ndarray,
        process_id: int,
        total_processes: int
        ) -> np.ndarray:
        """
        Extract process-specific offsets from global offset array.
        
        Uses stride-based indexing for efficient distribution of requests
        across worker processes.
        
        Args:
            global_offsets: Array of cumulative time offsets
            process_id: Current process ID (0-indexed)
            total_processes: Total number of processes
            
        Returns:
            Subarray of offsets assigned to the specified process
        """
        # Stride-based indexing: [start:stop:step]
        return global_offsets[process_id::total_processes]


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
        self.rps_plot_path = os.path.join(output_json_filepath, output_json_filename)

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
