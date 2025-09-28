import sys
import time
import struct
from queue import Empty
from typing import Dict
from multiprocessing import Event, shared_memory, BoundedSemaphore

import numpy as np
import multiprocessing as mp
import psutil
from tqdm import tqdm
from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.logging import get_logger

STATUS_REPORT_INTERVAL = 1
# Message queue format for communication with subprocesses: 5 integers.
# The 5 integers represent status, post, recv, fail, and finish respectively.
FMT = "5I"
MESSAGE_SIZE = struct.calcsize(FMT)

logger = get_logger()


def create_message_share_memory():
    """Create shared memory for inter-process communication.
    
    Returns:
        shared_memory.SharedMemory: Shared memory object for message passing.
    """
    shm = shared_memory.SharedMemory(create=True, size=MESSAGE_SIZE)
    buf = shm.buf
    # Set flag to 2, indicating child process is ready for first batch data deserialization
    buf[:] = struct.pack(FMT, 2, 0, 0, 0, 0)
    return shm


def check_virtual_memory_usage(dataset_bytes, threshold_percent=80):
    """Check current virtual memory usage and raise exception if threshold is exceeded.

    Uses psutil library for cross-platform memory monitoring.

    Args:
        dataset_bytes (int): Dataset size in bytes
        threshold_percent (int): Memory usage threshold percentage, default 80%

    Raises:
        MemoryError: When virtual memory usage exceeds threshold
    """
    try:
        # Get memory information using psutil
        memory = psutil.virtual_memory()
        
        # Extract memory information (all values are in bytes)
        total_mem = memory.total
        available_mem = memory.available
        used_mem = memory.used

        # Calculate memory usage after adding dataset
        total_used_after_dataset = used_mem + dataset_bytes
        usage_percent = (total_used_after_dataset / total_mem) * 100 if total_mem > 0 else 0
        
        # Check if usage exceeds threshold
        if usage_percent > threshold_percent:
            error_msg = (
                f"Virtual memory usage too high: {usage_percent:.2f}% > {threshold_percent}% "
                f"(Total memory: {total_mem / (1024**3):.2f} GB, "
                f"Used: {used_mem / (1024**3):.2f} GB, "
                f"Available: {available_mem / (1024**3):.2f} GB, "
                f"Dataset size: {dataset_bytes / (1024**2):.2f} MB)"
            )
            logger.error(error_msg)
            raise MemoryError(error_msg)
        
        logger.info(f"Serialized dataset size: {dataset_bytes / (1024**2):.2f} MB")
        logger.info(
            f"Memory usage check passed: {usage_percent:.2f}% < {threshold_percent}% "
            f"(Available: {available_mem / (1024**3):.2f} GB)"
        )

    except Exception as e:
        logger.warning(f"Error occurred while checking virtual memory usage: {e}")
        raise


class ProgressBar:
    """Progress monitor reading per-worker SharedMemory objects.

    Args:
        per_pid_shms: Mapping from worker pid to SharedMemory instance
        stop_event: Event to signal when to stop monitoring
        data_num: Total number of data items to process
        debug: Whether to run in debug mode
        pressure: Whether to run in pressure testing mode
        refresh_interval: Interval for refreshing progress display
    """

    def __init__(
        self,
        per_pid_shms: Dict[int, shared_memory.SharedMemory],
        stop_event: Event,
        data_num: int = -1,
        debug: bool = False,
        pressure: bool = False,
        refresh_interval: float = 1.0,
    ):
        self.debug = debug
        self.stop_event = stop_event
        self.data_num = data_num
        self.logger = get_logger()

        # expected: pid -> SharedMemory instance
        # We copy the mapping so external mutations are allowed but won't break internal dict ops.
        self.per_pid_shms: Dict[int, shared_memory.SharedMemory] = per_pid_shms

        self.pressure = pressure
        self.refresh_interval = refresh_interval

        self.per_pid_stats: Dict[int, Dict[str, int]] = {}
        self.stats = {"post": 0, "recv": 0, "fail": 0, "finish": 0}
        self._keys = ("post", "recv", "fail", "finish")

        self.start_time = time.perf_counter()
        self._last_snapshot_time = self.start_time
        self._last_snapshot_stats = self.stats.copy()

    # ------------------- aggregation logic -------------------
    def _recalc_aggregate(self):
        """Recalculate aggregate statistics from per-pid stats."""
        agg = {k: 0 for k in self._keys}
        for st in self.per_pid_stats.values():
            for k in self._keys:
                agg[k] += int(st.get(k, 0))
        self.stats = agg

    def _read_shared_memory_and_update_per_pid(self) -> bool:
        """Read shared memory and update per-pid statistics.
        
        Returns:
            bool: True if any per-pid stat changed, False otherwise.
        """
        updated = False
        # Iterate over a snapshot of keys to allow external mapping mutations
        for pid, shm in self.per_pid_shms.items():
            raw = bytes(shm.buf[:MESSAGE_SIZE])
            status, post, recv, fail, finish = struct.unpack(FMT, raw)
            normalized = {
                "post": max(0, int(post)),
                "recv": max(0, int(recv)),
                "fail": max(0, int(fail)),
                "finish": max(0, int(finish)),
            }
            prev = self.per_pid_stats.get(pid)
            if prev != normalized:
                self.per_pid_stats[pid] = normalized
                updated = True
        if updated:
            self._recalc_aggregate()
        return updated

    # ------------------- rate computations -------------------
    def _compute_rates_since_start(self):
        """Compute rates since the start of monitoring."""
        now = time.perf_counter()
        dt = max(1e-6, now - self.start_time)
        return {k: self.stats.get(k, 0) / dt for k in self._keys}

    def _compute_rates_interval(self):
        """Compute rates for the current interval."""
        now = time.perf_counter()
        dt = now - self._last_snapshot_time
        if dt <= 0:
            return {k: 0.0 for k in self._keys}
        rates = {}
        for k in self._keys:
            rates[k] = (self.stats.get(k, 0) - self._last_snapshot_stats.get(k, 0)) / dt
        self._last_snapshot_time = now
        self._last_snapshot_stats = self.stats.copy()
        return rates

    def _format_per_pid_brief(self) -> str:
        """Format per-pid statistics into a brief string."""
        items = []
        for pid, st in sorted(self.per_pid_stats.items()):
            finish = st.get("finish", 0)
            post = st.get("post", 0)
            recv = st.get("recv", 0)
            fail = st.get("fail", 0)
            items.append(f"{pid}:{finish}/{post}/{recv}/{fail}")
        if not items:
            return "<no workers>"
        return " | ".join(items)

    # ---------- normal: two-line fixed display ----------
    def _draw_progress(self):
        """Draw progress bar with statistics."""
        if self.data_num <= 0:
            raise ValueError("Data num must be greater than 0 for progress bar display")
        self.logger.info(f"Starting progress bar, data num: {self.data_num}")
        # leave=True ensures final display is retained after closing
        main_bar = tqdm(
            total=self.data_num, desc="Progress", unit="req", position=0, leave=True
        )
        info_bar = tqdm(total=1, desc="", bar_format="{desc}", position=1, leave=True)

        try:
            initial = min(
                int(self.stats.get("finish", 0) or self.stats.get("post", 0)),
                self.data_num,
            )
            if initial > 0:
                main_bar.update(initial)

            last_update = 0.0
            while main_bar.n <= self.data_num and not self.stop_event.is_set():
                updated = self._read_shared_memory_and_update_per_pid()
                if updated:
                    new_count = min(
                        int(self.stats.get("finish", 0)),
                        self.data_num,
                    )
                    if new_count > main_bar.n:
                        main_bar.update(new_count - main_bar.n)

                now = time.perf_counter()
                if now - last_update >= self.refresh_interval:
                    rates = self._compute_rates_interval()
                    info = (
                        f"POST={self.stats['post']} ({rates['post']:.1f}/s)  "
                        f"RECV={self.stats['recv']} ({rates['recv']:.1f}/s)  "
                        f"FAIL={self.stats['fail']} ({rates['fail']:.1f}/s)  "
                        f"FIN={self.stats['finish']} ({rates['finish']:.1f}/s)   "
                    )
                    info_bar.set_description_str(info)
                    info_bar.refresh()
                    last_update = now

                time.sleep(min(0.2, self.refresh_interval))
        except KeyboardInterrupt:
            pass
        finally:
            self._read_shared_memory_and_update_per_pid()
            new_count = min(
                int(self.stats.get("finish", 0)),
                self.data_num,
            )
            main_bar.update(new_count - main_bar.n)
            rates = self._compute_rates_interval()
            info = (
                f"POST={self.stats['post']} ({rates['post']:.1f}/s)  "
                f"RECV={self.stats['recv']} ({rates['recv']:.1f}/s)  "
                f"FAIL={self.stats['fail']} ({rates['fail']:.1f}/s)  "
                f"FINISH={self.stats['finish']} ({rates['finish']:.1f}/s)   "
            )
            info_bar.set_description_str(info)
            info_bar.refresh()

            main_bar.close()
            info_bar.close()

    # ---------- pressure: single-shot print ----------
    def print_pressure_snapshot(self, end_with_newline: bool = False):
        """Print a snapshot of pressure test statistics.
        
        Args:
            end_with_newline: Whether to end with a newline character
        """
        self._read_shared_memory_and_update_per_pid()
        rates = self._compute_rates_since_start()
        summary = (
            f"POST={self.stats['post']} ({rates['post']:.1f}/s)  "
            f"RECV={self.stats['recv']} ({rates['recv']:.1f}/s)  "
            f"FAIL={self.stats['fail']} ({rates['fail']:.1f}/s)  "
            f"FIN={self.stats['finish']} ({rates['finish']:.1f}/s)"
        )
        out = f"{summary}"
        if end_with_newline:
            sys.stdout.write("\r" + out + "\n")
        else:
            sys.stdout.write("\r" + out)
        sys.stdout.flush()

    def _draw_pressure_messages(self):
        """Draw pressure test messages with periodic updates."""
        last_print_t = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                self._read_shared_memory_and_update_per_pid()
                now = time.perf_counter()
                if now - last_print_t >= self.refresh_interval:
                    self.print_pressure_snapshot(end_with_newline=False)
                    last_print_t = now
                time.sleep(min(0.2, self.refresh_interval))
        except KeyboardInterrupt:
            pass
        finally:
            self.print_pressure_snapshot(end_with_newline=True)

    def _refresh_task_monitor(self, task_state_manager: TaskStateManager):
        """Refresh task monitor with current statistics.
        
        Args:
            task_state_manager: Task state manager for updating status
        """
        if not self.pressure:
            task_state_manager.update_task_state(
                {
                    "total_count": self.data_num,
                }
            )
        while not self.stop_event.is_set():
            updated = self._read_shared_memory_and_update_per_pid()
            if updated:
                rates = self._compute_rates_interval()
                finish_rate = round(rates["finish"], 1)
                state = {
                    "status": "inferencing",
                    "finish_count": self.stats["finish"],
                    "other_kwargs": {
                        "POST": self.stats["post"],
                        "RECV": self.stats["recv"],
                        "FAIL": self.stats["fail"],
                    },
                }
                if finish_rate > 0:
                    state["progress_description"] = f"[{finish_rate} it/s]"
                task_state_manager.update_task_state(state)
            time.sleep(STATUS_REPORT_INTERVAL)
        self._read_shared_memory_and_update_per_pid()
        state = {
            "status": "write cache",
            "finish_count": self.stats["finish"],
            "other_kwargs": {
                "POST": self.stats["post"],
                "RECV": self.stats["recv"],
                "FAIL": self.stats["fail"],
            },
        }
        task_state_manager.update_task_state(state)

    def set_message_flag(self, flag: int):
        """Set message flag for all shared memory objects.
        
        Args:
            flag: Flag value to set
        """
        for _, shm in self.per_pid_shms.items():
            shm.buf[:MESSAGE_SIZE] = struct.pack(FMT, flag, 0, 0, 0, 0)

    def display(self, task_state_manager: TaskStateManager):
        """Display progress monitoring.
        
        Args:
            task_state_manager: Task state manager for updating status
        """
        while self.stop_event.is_set():
            need_wait = any(
                struct.unpack_from(FMT, shm.buf, 0)[0] == 2
                for shm in self.per_pid_shms.values()
            )
            if not need_wait:
                self.logger.info(
                    "All subprocesses have finished deserializing the first batch of data"
                )
                self.stop_event.clear()
                break
            time.sleep(STATUS_REPORT_INTERVAL)
        if not self.debug:
            self._refresh_task_monitor(task_state_manager)

        if self.debug:
            if self.pressure:
                self._draw_pressure_messages()
            else:
                self._draw_progress()


class TokenProducer:
    """Token generator for controlling request pacing in multi-process scenarios.
    
    Produces tokens according to request_rate and optional traffic_cfg to control
    multi-process request pacing.
    """

    def __init__(
        self,
        request_rate: int,
        traffic_cfg: ConfigDict,
        request_num: int = None,
        pressure_mode: bool = False,
    ):
        """
        Args:
            request_rate: Desired request rate (RPS) used to pace requests.
            traffic_cfg: Traffic configuration controlling ramp-up and burstiness.
            request_num: Total number of requests to schedule when known.
            pressure_mode: If True, after generating the first `request_num` tokens
                (used to warm up connections), subsequent tokens are produced without sleep.
        """
        self.logger = get_logger()
        self.request_rate = request_rate
        self.pressure_mode = pressure_mode
        self.burstiness = 1.0
        # When request_rate < 0.1, treat as infinite (no pacing applied here)
        if self.request_rate < 0.1:
            self.request_rate = "INF"
            self.token_bucket = None
        else:
            self.token_bucket = BoundedSemaphore(request_num + 1)
            # First release all tokens in token_bucket to make it empty
            for _ in range(request_num + 1):
                self.token_bucket.acquire()

        # print(f"Token producer init, {self.token_bucket}")

        # If `traffic_cfg` is provided, pre-generate `interval_lists` for ramp-up; after
        # exhausting it, fall back to gamma-distributed intervals based on request_rate.
        self.interval_lists = []
        if traffic_cfg:
            self.burstiness = float(traffic_cfg.get("burstiness", self.burstiness))
            ramp_up_strategy = traffic_cfg.get("ramp_up_strategy")
            ramp_up_start_rps = traffic_cfg.get("ramp_up_start_rps")
            ramp_up_end_rps = traffic_cfg.get("ramp_up_end_rps")
            if ramp_up_strategy:
                self.logger.info(
                    f"Traffic ramp-up strategy: {ramp_up_strategy}. Will increase "
                    f"RPS from {ramp_up_start_rps} to {ramp_up_end_rps} RPS over "
                    "the duration of the benchmark."
                )
                # TODO check traffic_cfg
                self.interval_lists = self._generate_interval_lists(
                    request_num,
                    self.burstiness,
                    ramp_up_strategy,
                    ramp_up_start_rps,
                    ramp_up_end_rps,
                )
            else:
                self.logger.info(
                    f"Traffic request rate: {request_rate} RPS with burstiness {self.burstiness}."
                )
        else:
            self.logger.info(f"Traffic request rate: {request_rate} RPS.")

    def _generate_interval_lists(
        self,
        request_num: int,
        burstiness: float,
        ramp_up_strategy: str,
        ramp_up_start_rps: int,
        ramp_up_end_rps: int,
    ):
        """Generate interval lists for request pacing.
        
        Args:
            request_num: Total number of requests
            burstiness: Burstiness factor for request distribution
            ramp_up_strategy: Strategy for ramping up requests (linear/exponential)
            ramp_up_start_rps: Starting RPS for ramp-up
            ramp_up_end_rps: Ending RPS for ramp-up
            
        Returns:
            List of sleep intervals for request pacing
        """
        # Precompute delays among requests to minimize request send lag
        interval_lists = []
        delay_ts = []
        for request_index in range(request_num):
            progress = request_index / max(request_num - 1, 1)
            if ramp_up_strategy == "linear":
                increase = (ramp_up_end_rps - ramp_up_start_rps) * progress
                current_request_rate = ramp_up_start_rps + increase
            else:  # exponential
                ratio = ramp_up_end_rps / ramp_up_start_rps
                current_request_rate = ramp_up_start_rps * (ratio**progress)
            if current_request_rate == float("inf"):
                delay_ts.append(0)
            else:
                theta = 1.0 / (current_request_rate * burstiness)

                # Sample the request interval from the gamma distribution
                # If burstiness is 1, it follows exponential distribution
                delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))

        # Calculate the cumulative delay time from the first sent requests
        for i in range(1, len(delay_ts)):
            delay_ts[i] += delay_ts[i - 1]
        if ramp_up_strategy is None and delay_ts[-1] != 0:
            # When ramp_up_strategy is not set, assume fixed request rate
            # and scale delay time to align with target_total_delay_s
            #
            # NOTE: Accumulating random delta values from gamma distribution
            # would have 1-2% gap from target_total_delay_s. This logic
            # closes the gap to stabilize throughput data across different seeds
            target_total_delay_s = request_num / self.request_rate
            normalize_factor = target_total_delay_s / delay_ts[-1]
            delay_ts = [delay * normalize_factor for delay in delay_ts]

        start_ts = time.time()
        for request_index in range(request_num):
            if delay_ts[request_index] > 0:
                current_ts = time.time()
                sleep_interval_s = start_ts + delay_ts[request_index] - current_ts
                if sleep_interval_s > 0:
                    interval_lists.append(sleep_interval_s)
        return interval_lists

    def produce_token(self, stop_evt: Event):
        """Produce tokens for request pacing.
        
        Args:
            stop_evt: Event to signal when to stop token production
        """
        if not self.token_bucket:
            return

        # Wait for child process to complete first batch data loading
        while stop_evt.is_set():
            time.sleep(0.5)
        self.logger.info("Token producer started")
        interval_index = 0
        theta = 1.0 / (self.request_rate * self.burstiness)

        while not stop_evt.is_set():
            if interval_index < len(self.interval_lists):
                interval = self.interval_lists[interval_index]
                self.token_bucket.release()
                time.sleep(interval)
                interval_index += 1
            else:
                try:
                    # After first batch requests are sent, subsequent requests
                    # are not sent according to request rate strategy
                    self.token_bucket.release()

                except ValueError as e:
                    # ValueError: semaphore or lock released too many times
                    # Indicates token bucket is full, wait for tokens to be used
                    interval = np.random.gamma(shape=self.burstiness, scale=theta)
                    time.sleep(interval)
