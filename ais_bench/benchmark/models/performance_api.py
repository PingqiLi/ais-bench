import os
import time
import uuid
import multiprocessing
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm

from ais_bench.benchmark.utils.results import MiddleData
from ais_bench.benchmark.utils.tokenizer import BenchmarkTokenizer
from ais_bench.benchmark.models.base_api import BaseAPIModel


class PerformanceAPIModel(BaseAPIModel):
    def __init__(
        self,
        path: str,
        request_rate: int = 1,
        rpm_verbose: bool = False,
        retry: int = 2,
        max_seq_len: int = 2048,
        meta_template: Optional[Any] = None,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(
            path,
            request_rate,
            rpm_verbose,
            retry,
            max_seq_len,
            meta_template,
            generation_kwargs,
            verbose,
        )
        self.path = path
        self.do_performance = False
        self.client = None
        self.tokenizer: Optional[BenchmarkTokenizer] = None
        self.result_cache: Dict[str, MiddleData] = defaultdict(MiddleData)

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

    def prepare_input_data(self, input_text: str, data_id:int = -1) -> MiddleData:
        """Prepare input data, tokenize if performance mode is enabled."""
        rrid = uuid.uuid4().hex
        cache_data = self.result_cache[rrid]
        cache_data.request_id = rrid
        cache_data.data_id = data_id
        cache_data.input_data = input_text
        if self.do_performance and self.tokenizer:
            time_cost, token_id = self.encode(input_text)
            cache_data.input_token_id = token_id
            cache_data.num_input_tokens = len(token_id)
            cache_data.num_input_chars = len(input_text)
        return cache_data

    def set_result(self, data: MiddleData) -> None:
        """Update decoding information for a given request."""
        if not data.output:
            self.logger.warning(
                f"Request {data.request_id} has no output. Please check the server response."
            )
            data.is_empty = True
        data.is_success = True

    def encode(self, prompt: str) -> Tuple[float, List[int]]:
        """Encode a string into tokens, measuring processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return 0.0, []
        if not isinstance(prompt, str):
            self.logger.warning(f"Input type: expected a string, got {type(prompt)}, InputTokens will be 0.")
            return 0.0, []
        time_start = time.perf_counter()
        tokens = self.tokenizer.encode(prompt)
        time_cost = (time.perf_counter() - time_start) * 1000  # Convert to milliseconds
        return time_cost, tokens

    def decode(self, tokens: List[int]) -> Tuple[List[float], str]:
        """Decode tokens into a string, measuring processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return [], ""

        time_start = time.perf_counter()
        prompt = self.tokenizer.decode(tokens)
        time_cost = [(time.perf_counter() - time_start) * 1000]  # Convert to milliseconds
        return time_cost, prompt

    def decode_stream(self, tokens: List[int]) -> Tuple[List[float], List[str]]:
        """Decode tokens into a string stream, measuring per-token processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return [], []

        prompt = []
        time_cost = []
        time_start = time.perf_counter()
        for token in tokens:
            prompt.append(self.tokenizer.decode(token))
            time_cost.append(
                (time.perf_counter() - time_start) * 1000
            )  # Convert to milliseconds
            time_start = time.perf_counter()
        return time_cost, prompt

    def get_performance_data(self) -> List[Dict[str, Any]]:
        """Retrieve performance data from cached results."""
        if self.do_performance:
            if self.tqdm_pos < 0:
                pos = None
            else:
                pos = 3 * self.tqdm_pos + 2
            for key, _ in tqdm(self.result_cache.items(), desc="Encoding output text...", position=pos, total=len(self.result_cache)):
                # Failed requests are not saved
                if not self.result_cache[key].is_success:
                    continue
                time_cost, tokens = self.encode(self.result_cache[key].output)
                self.result_cache[key].num_generated_tokens = len(tokens)
        performance_data = []
        try:
            performance_data = [
                cache_data.convert_to_performance_data()
                for cache_data in self.result_cache.values()
            ]
        except Exception as e:
            self.logger.error(f"Error converting performance data: {e}")
        finally:
            self.result_cache.clear()
        return performance_data
