import csv
import json
import copy
import orjson
import h5py
from tqdm import tqdm
import numpy as np
import collections
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
import copy
from ais_bench.benchmark.utils import get_logger

MAX_H5_CHUNK_SIZE = 5000


def dump_results_dict(results_dict, filename, formatted = True):
    with open(filename, 'w', encoding='utf-8') as json_file:
        if formatted:
            json.dump(results_dict, json_file, indent=4, ensure_ascii=False)
        else:
            json.dump(results_dict, json_file, ensure_ascii=False)

def fast_dump_results_dict(results_dict, filename):
    with open(filename, 'wb') as f:
        f.write(orjson.dumps(results_dict))


def dump_list_as_h5(data_list, h5_file, data_type=np.float32):
    total_rows = len(data_list)
    chunk_size = MAX_H5_CHUNK_SIZE if total_rows >= MAX_H5_CHUNK_SIZE else total_rows
    with h5py.File(h5_file, 'w') as f:
        dt = h5py.vlen_dtype(data_type)
        dset = f.create_dataset('arrays', (total_rows,), dtype=dt,
                            compression='gzip', chunks=(chunk_size,))

        # 分块写入（带进度条）
        for i in tqdm(range(0, total_rows, chunk_size), desc="Dumping data to h5"):
            end_idx = min(i + chunk_size, total_rows)
            chunk = data_list[i:end_idx]
            for j, arr in enumerate(chunk):
                dset[i+j] = arr


def load_from_h5(h5_file):
    with h5py.File(h5_file, 'r') as f:
        dset = f['arrays']
        data = dset[:]
    return data


@dataclass
class MiddleData:
    data_id: int = -1
    input_data: Optional[str] = None
    input_token_id: list[int] = field(default_factory=list)
    num_input_tokens: int = 0
    num_input_chars: int = 0
    num_generated_tokens: int = 0
    num_generated_chars: int = 0
    prefill_latency: float = 0.0
    decode_cost: list[float] = field(default_factory=list)
    req_latency: float = 0.0
    decode_batch_size: list[int] = field(default_factory=list)
    prefill_batch_size: int = 0
    post_process_time: float = 0.0
    queue_wait_time: list[float] = field(default_factory=list)
    data_option: list = field(default_factory=list)
    output: str = ""
    output_reasoning: str = ""
    output_token_id: list[int] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    request_id: str = ""
    model_id: str = ""
    is_success: bool = False
    is_empty: bool = False
    multiturn_group_id: str = ""

    def is_valid(self):
        """To ensure valid results"""
        return all(
            [
                self.num_input_tokens,
                self.num_input_chars,
                self.num_generated_tokens,
                self.num_generated_chars,
                self.decode_cost,
            ]
        )


    def get_output(self) -> str:
        if self.output_reasoning:
            if self.output:
                return self.output_reasoning + '</think>' + self.output
            else:
                return self.output_reasoning
        else:
            return self.output


    def convert_to_performance_data(self) -> dict:
        converted_data = {
            "id": self.data_id,
            "input_data": self.input_data,
            "input_token_id": self.input_token_id,
            "output": self.get_output(),
            "output_token_id": self.output_token_id,
            "prefill_latency": self.prefill_latency,
            "prefill_throughput": 0,
            "decode_token_latencies": self.decode_cost,
            "last_decode_latency": 0.0,
            "decode_max_token_latency": 0.0,
            "seq_latency": self.req_latency,
            "input_tokens_len": self.num_input_tokens,
            "generate_tokens_len": self.num_generated_tokens,
            "generate_tokens_speed": 0,
            "input_characters_len": len(self.input_data),
            "generate_characters_len": self.num_generated_chars,
            "characters_per_token": 0.0,
            "prefill_batch_size": self.prefill_batch_size,
            "decode_batch_size": self.decode_batch_size[:],
            "queue_wait_time": self.queue_wait_time[:],
            "request_id": self.request_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_success": self.is_success,
            "is_empty": self.is_empty,
            "multiturn_group_id": self.multiturn_group_id
        }
        if self.is_success: # calculate when request succeed
            converted_data["prefill_throughput"] = round(len(self.input_token_id) / self.prefill_latency * 1000, 4) if self.prefill_latency > 0 else 0
            converted_data["last_decode_latency"] = float(self.decode_cost[-1]) if self.decode_cost.any() else 0.0
            converted_data["decode_max_token_latency"] = float(np.max(self.decode_cost)) if self.decode_cost.any() else 0.0
            converted_data["generate_tokens_speed"] = round(self.num_generated_tokens / self.req_latency * 1000, 4) if self.req_latency > 0 else 0
            converted_data["characters_per_token"] = (
                round(self.num_generated_chars / self.num_generated_tokens, 4)
                if self.num_generated_tokens
                else 0.0
            )

        return converted_data
