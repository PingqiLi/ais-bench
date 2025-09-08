import inspect
import json
import os
import os.path as osp
import time
import multiprocessing
from multiprocessing import RLock, freeze_support
from typing import List, Optional, Tuple

import mmengine
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from mmengine.config import ConfigDict

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.utils import batched
from ais_bench.benchmark.utils.build import build_model_from_cfg
from ais_bench.benchmark.global_consts import WORKERS_NUM
from ..utils.logging import get_logger
from .icl_base_inferencer import GenInferencerOutputHandler
from ais_bench.benchmark.utils.results import dump_results_dict, fast_dump_results_dict, dump_list_as_h5
from .icl_gen_perf_inferencer import GenPerfInferencer
from .icl_gen_inferencer import DEFAULT_MAX_CONCURRENCY_PER_PROCESS

logger = get_logger(__name__)

def pressure_single_model(
    model_cfg, shared_inputs, lock, total_thread_count, total_input_idx, **extra_gen_kwargs
):
    model = build_model_from_cfg(model_cfg)
    if extra_gen_kwargs.get("is_synthetic"):
        model.set_synthetic()
    if extra_gen_kwargs.get("do_performance"):
        model.set_performance()
    model.pressure_generate_from_queue(
        shared_inputs,
        lock,
        total_thread_count,
        total_input_idx,
        **extra_gen_kwargs,
    )
    if not hasattr(model, "set_performance"):
        raise AttributeError(f'{model} has no except outputs, please check model config')
    return model.get_performance_data()


@ICL_INFERENCERS.register_module()
class GenPressureInferencer(GenPerfInferencer):
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
            is_synthetic,
            num_prompts,
            **kwargs,
        )


    def pressure_infer_with_multiprocess(
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
        if isinstance(model_cfg, dict) and getattr(model_cfg, "traffic_cfg", {}):
            logger.warning("Traffic request rate control parameters are not supported in pressure inference scenarios, would be ignored.")

        logger.info(f"Concurrency is set to {max_concurrency}, infer with total {workers_num} process")
        q, r = divmod(max_concurrency, workers_num)
        concurrencys = [q + 1] * r + [q] * (workers_num - r)
        with multiprocessing.Manager() as manager:
            total_thread_count = manager.Value('t', 0)
            total_input_idx = manager.Value('i', 0)
            shared_inputs = manager.list(inputs)
            lock = manager.Lock()

            request_rate = model.request_rate
            if request_rate < 0.1:
                logger.info(f"get request_rate {request_rate} small than 0.1, all requests will send together!")
                request_rate = 0
            request_rate_mean = request_rate / workers_num
            # Set the timing of token release according to qps, only one request can hold the token at each moment
            freeze_support()
            pool = multiprocessing.Pool(processes=workers_num, initializer=tqdm.set_lock, initargs=(RLock(),))
            async_results = []
            for i in range(workers_num):
                new_gen_kwargs = extra_gen_kwargs.copy()
                new_gen_kwargs.update({
                    "concurrency": max(1, concurrencys[i]),
                    "total_concurrency": max_concurrency,
                    "process_id":  i,
                })
                res = pool.apply_async(func=pressure_single_model,
                                        args=(model_cfg, shared_inputs, lock, total_thread_count, total_input_idx),
                                        kwds=new_gen_kwargs,
                                        error_callback=lambda x:logger.error(x)
                                        )
                async_results.append(res)
            pool.close()
            pool.join()
            for res in async_results:
                results.extend(res.get())
        return results

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

        logger.info("Starting performance inference process...")
        start_time_stamp = time.perf_counter()

        # Prepare inference parameters
        extra_gen_kwargs = self._build_extra_gen_kwargs()
        # Run inference
        with torch.no_grad():
            parsed_entries = self.model.parse_template(entry, mode='gen')
            results = self.pressure_infer_with_multiprocess(
                self.model, self.model_cfg, parsed_entries, golds, **extra_gen_kwargs)
        preds = self.extract_preds(results)
        preds['id'] = [i for i in range(len(preds['request_id']))]
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
                osp.join(output_filepath, output_filename + "_details.json"),
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

