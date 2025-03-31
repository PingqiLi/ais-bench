import csv
import collections
from dataclasses import dataclass, field

import numpy as np
import copy
from ais_bench.benchmark.utils import get_logger


@dataclass
class MiddleData:
    data_id: str = ""
    input_data: str = ""
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
    tokenized_time: float = 0.0
    detokenized_time: list[float] = field(default_factory=list)
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
            * 1000,
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
            * 1000,
            "input_characters_len": len(self.input_data),
            "generate_characters_len": self.num_generated_chars,
            "tokenizer_time": self.tokenized_time,
            "detokenizer_time": self.detokenized_time,
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
        }


class MetricsCalculator:
    def __init__(self, result: dict):
        self.data_count = len(result["is_success"])
        self.decode_latencies = result["decode_token_latencies"]
        self.success_count = sum(result["is_success"])
        self.empty_count = sum(result["is_empty"])
        self.infer_time = max(result["end_time"]) - min(result["start_time"])
        per_request_avg_decode_time = []
        # Compute the average decode latency per request
        for values in self.decode_latencies:
            if values:  # Skip empty lists
                per_request_avg_decode_time.append(round(np.average(values), 4))
        result["average_decode_latencies"] = per_request_avg_decode_time[:]
        self.result = self.convert_result(copy.deepcopy(result))

        self.logger = get_logger()

        def new_metric_result():
            return {
                "Average": 0,
                "Max": 0,
                "Min": 0,
                "Median": 0,
                "P75": 0,
                "P90": 0,
                "P99": 0,
                "N": 0,
            }

        self.metrics = collections.defaultdict(new_metric_result)

        def new_common_result():
            return {"Value": 0}

        self.common_metrics = collections.defaultdict(new_common_result)

    def get_common_res(self, concurrency):
        self.common_metrics.update({"Max Concurrency": concurrency})
        return self.common_metrics

    def save_performance(self, out_path: str):
        """
        Save performance metrics to a CSV file.

        :param out_path: Path to the output CSV file.
        """
        if not self.metrics:
            raise ValueError("Metrics data is empty, cannot save to file.")

        try:
            # Extract headers from the first available entry
            first_entry = next(iter(self.metrics.values()), None)
            if first_entry is None:
                raise ValueError("Metrics data structure is invalid.")

            headers = list(first_entry.keys())

            with open(out_path, mode="w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)

                # Write header: First column is the object name, followed by metric keys
                writer.writerow(["Performance Parameters"] + headers)

                # Write each object's data
                for obj_name, values in self.metrics.items():
                    row = [obj_name] + [values.get(key, "") for key in headers]
                    writer.writerow(row)

        except (OSError, IOError) as e:
            raise RuntimeError(f"Failed to write to file '{out_path}': {e}")

    def convert_result(self, result: dict):
        remove_keys = [
            "id",
            "start_time",
            "end_time",
            "input_data",
            "input_token_id",
            "is_success",
            "is_empty",
            "request_id",
            "output",
            "output_token_id",
        ]
        for key in remove_keys:
            result.pop(key, None)
        mapping = {
            "seq_latency": "Latency",
            "prefill_latency": "TTFT",
            "average_decode_latencies": "TPOT",
            "decode_token_latencies": "ITL",
            "input_tokens_len": "InputTokens",
            "generate_tokens_len": "OutputTokens",
            "prefill_throughput": "PrefillTokenThroughput",
            "generate_tokens_speed": "OutputTokenThroughput",
            "tokenizer_time": "Tokenizer",
            "detokenizer_time": "Detokenizer",
            "prefill_batch_size": "PrefillBatchsize",
            "decode_batch_size": "DecoderBatchsize",
            "queue_wait_time": "QueueWaitTime",
        }

        ans = {mapping_value: [] for mapping_value in mapping.values()}

        # Use a dictionary comprehension to populate the values
        for mapping_key, mapping_value in mapping.items():
            for value in result[mapping_key]:
                if isinstance(value, list):
                    ans[mapping_value].extend(value)
                else:
                    ans[mapping_value].append(value)
        if sum(ans["PrefillBatchsize"]) == 0:
            ans.pop("PrefillBatchsize")

        for key in ["DecoderBatchsize", "QueueWaitTime"]:
            res = ans.get(key)
            if not res or not res[-1]:
                ans.pop(key)
        return ans

    def calculate(self):
        self.__calc_metrics()
        self.__calc_common_metrics()
        self.add_units()

    def __calc_metrics(self):
        """Calculate various statistical metrics for performance analysis."""
        # Iterate over all collected metrics
        for metric, value in self.result.items():
            stats = {
                "Average": 0,
                "Min": 0,
                "Max": 0,
                "Median": 0,
                "P75": 0,
                "P90": 0,
                "P99": 0,
            }

            if value:
                # Special handling for batch size metrics
                if metric in {"PrefillBatchsize", "DecoderBatchsize"}:
                    value = self.__statistic_prefill_or_decode_batch_size(value)

                # Compute statistical values
                stats["Average"] = round(np.average(value), 4)
                stats["Min"] = round(float(min(value)), 4)
                stats["Max"] = round(float(max(value)), 4)
                stats["Median"] = round(np.percentile(value, 50), 4)
                stats["P75"] = round(np.percentile(value, 75), 4)
                stats["P90"] = round(np.percentile(value, 90), 4)
                stats["P99"] = round(np.percentile(value, 99), 4)

            # Store the computed metrics
            self.metrics[metric] = stats

        # Assign fixed count value for all metrics
        for key in self.metrics:
            self.metrics[key]["N"] = self.success_count

    def __statistic_prefill_or_decode_batch_size(self, batch_sizes: list):
        """
        Process batch sizes by merging consecutive occurrences of the same number based on the first occurrence.

        Example:
            Input:  [2, 2, 5, 5, 3, 3, 5, 5, 5, 3]
            Output: [2, 5, 3]

        The method follows these rules:
        - When encountering a number X for the first time, store it and track its expected count (X-1).
        - Each subsequent occurrence of X decreases its count.
        - If count reaches zero, the number is removed from tracking.
        - If any unprocessed numbers remain, a warning is logged.

        Args:
            batch_sizes (list): List of batch sizes.

        Returns:
            list: Processed list of batch sizes.
        """
        if not batch_sizes:
            return []

        statistics = []
        count_dict = {}

        for batch_size in batch_sizes:
            if batch_size in count_dict:
                # Reduce remaining expected occurrences
                count_dict[batch_size] -= 1
                if count_dict[batch_size] == 0:
                    del count_dict[batch_size]  # Remove once count reaches zero
            else:
                # Register new batch size and track expected occurrences
                count_dict[batch_size] = batch_size - 1
                statistics.append(batch_size)

        if count_dict:
            self.logger.warning("Batch size is not fully compressed: %s", count_dict)

        return statistics

    def __calc_common_metrics(self):
        self.common_metrics["Benchmark Duration"] = round(self.infer_time, 4)
        self.common_metrics["Total Requeusts"] = self.data_count
        self.common_metrics["Failed Requests"] = self.data_count - self.success_count
        self.common_metrics["Success Requests"] = self.success_count
        self.common_metrics["Concurrency"] = round(
            sum(self.result["Latency"]) / self.infer_time / 1000, 4
        )
        self.common_metrics["Max Concurrency"] = self.common_metrics["Concurrency"]

        try:
            self.common_metrics["Request Throughput"] = round(
                self.success_count / self.infer_time, 4
            )
        except ZeroDivisionError:
            self.common_metrics["Request Throughput"] = 0

        self.common_metrics["Total Input Tokens"] = sum(self.result["InputTokens"])
        if self.common_metrics["Total Input Tokens"] != 0:
            self.common_metrics["Prefill Token Throughput"] = round(
                1000
                * self.common_metrics["Total Input Tokens"]
                / sum(self.result["TTFT"]),
                4,
            )
        else:
            self.common_metrics["Prefill Token Throughput"] = 0

        self.common_metrics["Total Output Tokens"] = sum(self.result["OutputTokens"])
        if self.infer_time > 0:
            self.common_metrics["Input Token Throughput"] = round(
                self.common_metrics["Total Input Tokens"] / self.infer_time, 4
            )
            self.common_metrics["Output Token Throughput"] = round(
                sum(self.result["OutputTokens"]) / self.infer_time, 4
            )
            self.common_metrics["Total Token Throughput"] = round(
                (
                    self.common_metrics["Total Input Tokens"]
                    + sum(self.result["OutputTokens"])
                )
                / self.infer_time,
                4,
            )

    def add_units(self):
        ms = " ms"
        unit_token = " token/s"
        metrics_units_map = {
            "Latency": ms,
            "TTFT": ms,
            "TPOT": ms,
            "ITL": ms,
            "InputTokens": None,
            "OutputTokens": None,
            "PrefillTokenThroughput": unit_token,
            "OutputTokenThroughput": unit_token,
            "Tokenizer": ms,
            "Detokenizer": ms,
            "PrefillBatchsize": None,
            "DecoderBatchsize": None,
            "QueueWaitTime": " μs",
        }

        for metric, values in self.metrics.items():
            if metric not in metrics_units_map or metrics_units_map.get(metric) is None:
                continue
            for key, val in values.items():
                if key == "N":
                    continue
                values[key] = str(val) + metrics_units_map.get(metric)
        common_metric_units_map = {
            "Benchmark Duration": ms,
            "Total Requests": None,
            "Failed Requests": None,
            "Success Requests": None,
            "Concurrency": None,
            "Max Concurrency": None,
            "Request Throughput": " req/s",
            "Total Input Tokens": None,
            "Prefill Token Throughput": unit_token,
            "Input Token Throughput": " s",
            "Total Output Tokens": None,
            "Output Token Throughput": unit_token,
            "Total Token Throughput": unit_token,
        }

        for metric, value in self.common_metrics.items():
            if (
                metric not in common_metric_units_map
                or common_metric_units_map.get(metric) is None
            ):
                continue
            self.common_metrics[metric] = str(value) + common_metric_units_map.get(
                metric
            )
