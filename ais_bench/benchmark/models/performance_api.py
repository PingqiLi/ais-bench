import os
import threading
import time
import uuid
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from ais_bench.benchmark.utils.results import MiddleData
from ais_bench.benchmark.utils.tokenizer import BenchmarkTokenizer
from ais_bench.benchmark.models.base_api import BaseAPIModel


class PerformanceAPIModel(BaseAPIModel):
    def __init__(
        self,
        path: str,
        query_per_second: int = 1,
        rpm_verbose: bool = False,
        retry: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Any] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            path,
            query_per_second,
            rpm_verbose,
            retry,
            max_seq_len,
            meta_template,
            generation_kwargs,
            verbose,
        )
        self.data_id = 0
        self.path = path
        self.do_performance = False
        self.tokenizer: Optional[BenchmarkTokenizer] = None
        self.result_cache: Dict[str, MiddleData] = defaultdict(MiddleData)
        self.lock = threading.Lock()

    def set_performance(self) -> None:
        """Initialize the tokenizer and enable performance mode."""
        if self.path and os.path.exists(self.path):
            self.tokenizer = BenchmarkTokenizer(self.path)
            self.do_performance = True
            self.client.set_performance()
        else:
            raise ValueError(
                f"Tokenizer path '{self.path}' does not exist. "
                "Please set path in model config if you want to do performance infer"
            )

    def prepare_input_data(self, input_text: str) -> MiddleData:
        """Prepare input data, tokenize if performance mode is enabled."""
        rrid = uuid.uuid4().hex
        cache_data = self.result_cache[rrid]
        with self.lock:
            cache_data.data_id = str(self.data_id)
            self.data_id += 1
        cache_data.request_id = rrid
        cache_data.input_data = input_text

        if self.do_performance and self.tokenizer:
            time_cost, token_id = self.encode(input_text)
            cache_data.tokenized_time = time_cost
            cache_data.input_token_id = token_id
            cache_data.num_input_tokens = len(token_id)
            cache_data.num_input_chars = len(input_text)

        return cache_data

    def update_decode(self, data: MiddleData, stream: bool = True) -> None:
        """Update decoding information for a given request."""
        if not data.output:
            self.logger.warning(
                f"Request {data.request_id} has no output. Please check the server response."
            )
            data.is_empty = True
            return

        if not self.do_performance or not self.tokenizer:
            return

        _, data.output_token_id = self.encode(data.output)

        if stream:
            data.detokenized_time, _ = self.decode_stream(data.output_token_id)
        else:
            data.detokenized_time, _ = self.decode(data.output_token_id)

        data.is_success = True

    def encode(self, prompt: str) -> Tuple[float, List[int]]:
        """Encode a string into tokens, measuring processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return 0.0, []

        time_start = time.time()
        tokens = self.tokenizer.encode(prompt)
        time_cost = (time.time() - time_start) * 1000  # Convert to milliseconds
        return time_cost, tokens

    def decode(self, tokens: List[int]) -> Tuple[List[float], str]:
        """Decode tokens into a string, measuring processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return [], ""

        time_start = time.time()
        prompt = self.tokenizer.decode(tokens)
        time_cost = [(time.time() - time_start) * 1000]  # Convert to milliseconds
        return time_cost, prompt

    def decode_stream(self, tokens: List[int]) -> Tuple[List[float], List[str]]:
        """Decode tokens into a string stream, measuring per-token processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return [], []

        prompt = []
        time_cost = []
        time_start = time.time()
        for token in tokens:
            prompt.append(self.tokenizer.decode(token))
            time_cost.append(
                (time.time() - time_start) * 1000
            )  # Convert to milliseconds
            time_start = time.time()
        return time_cost, prompt

    def get_performance_data(self) -> List[Dict[str, Any]]:
        """Retrieve performance data from cached results."""
        performance_data = [
            cache_data.convert_to_performance_data()
            for cache_data in self.result_cache.values()
        ]
        total_count = len(self.result_cache)
        success_count = len(performance_data)

        if total_count != success_count:
            self.logger.warning(
                f"Total {total_count} requests, {success_count} returned."
            )

        self.result_cache.clear()
        return performance_data
