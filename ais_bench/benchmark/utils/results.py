import csv
import collections
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import numpy as np
import copy
from ais_bench.benchmark.utils import get_logger


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
    output_token_id: list[int] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    request_id: str = ""
    model_id: str = ""
    is_success: bool = False
    is_empty: bool = False
    chunk_time_point_list: list[int] = field(default_factory=list)

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

    def convert_to_performance_data(self) -> dict:
        return {
            "id": self.data_id,
            "input_data": self.input_data,
            "input_token_id": self.input_token_id,
            "output": self.output,
            "output_token_id": self.output_token_id,
            "prefill_latency": self.prefill_latency,
            "prefill_throughput": len(self.input_token_id)
            / self.prefill_latency
            * 1000 if self.prefill_latency > 0 else 0,
            "decode_token_latencies": self.decode_cost[:],
            "last_decode_latency": self.decode_cost[-1] if self.decode_cost else 0.0,
            "decode_max_token_latency": (
                max(self.decode_cost) if self.decode_cost else 0.0
            ),
            "seq_latency": self.req_latency,
            "input_tokens_len": self.num_input_tokens,
            "generate_tokens_len": self.num_generated_tokens,
            "generate_tokens_speed": self.num_generated_tokens
            / self.req_latency
            * 1000 if self.req_latency > 0 else 0,
            "input_characters_len": len(self.input_data),
            "generate_characters_len": self.num_generated_chars,
            "characters_per_token": (
                self.num_generated_chars / self.num_generated_tokens
                if self.num_generated_tokens
                else 0.0
            ),
            "prefill_batch_size": self.prefill_batch_size,
            "decode_batch_size": self.decode_batch_size[:],
            "queue_wait_time": self.queue_wait_time[:],
            "request_id": self.request_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "is_success": self.is_success,
            "is_empty": self.is_empty,
            "chunk_time_point_list": self.chunk_time_point_list
        }