import asyncio
import concurrent.futures
import os
import pickle
import queue as std_queue
import struct
import threading
import time
import uuid
from abc import abstractmethod
from multiprocessing import BoundedSemaphore, Queue, shared_memory
from typing import Any, Dict, Optional, Tuple

import aiohttp
import janus

from ais_bench.benchmark.global_consts import MAX_CHUNK_SIZE, REQUEST_TIME_OUT
from ais_bench.benchmark.openicl.utils import get_logger
from ais_bench.benchmark.tasks.utils import STATUS_REPORT_INTERVAL
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer import BaseInferencer

BLOCK_INTERVAL = 0.005  # Avoid request burst accumulation when RR is not configured
MAX_BATCH_SIZE = 100000  # Maximum concurrency
logger = get_logger(__name__)


class BaseApiInferencer(BaseInferencer):
    """Base Inferencer class for all evaluation Inferencer.

    Attributes mirroring your original class. This refactor keeps external API
    stable while avoiding blocking the asyncio event loop.
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        # Base class handles batch_size validation, model construction, output_handler initialization, etc.
        super().__init__(model_cfg, batch_size, output_json_filepath)

        self.save_every = max(1, int(save_every) if save_every is not None else 1)

        # Mode identification
        self.pressure_mode = mode == "pressure"
        self.perf_mode = mode == "perf" or self.pressure_mode

        # status_counter: If perf mode requires additional threads/counters, consider lazy creation to reduce overhead in normal mode.
        self.status_counter = StatusCounter()

    def _monitor_status_thread(
        self,
        stop_event: threading.Event,
        message_share_memory: shared_memory.SharedMemory,
    ) -> None:
        """Monitor status thread for reporting statistics.
        
        Args:
            stop_event: Event to signal thread termination
            message_share_memory: Shared memory for status communication
        """
        message_buf = message_share_memory.buf

        while not stop_event.is_set():
            post_req = self.status_counter.post_req
            get_req = self.status_counter.get_req
            failed_req = self.status_counter.failed_req
            finish_req = self.status_counter.finish_req

            # Pack -> bytes, then write back to corresponding slice of shared memory
            packed = struct.pack("<4I", post_req, get_req, failed_req, finish_req)
            # Write to 16 bytes starting from offset=4 (4 unsigned ints)
            flag = struct.unpack_from("<I", message_buf, 0)[0]
            if flag == 1:
                stop_event.set()
                break
            message_buf[4 : 4 + len(packed)] = packed
            time.sleep(STATUS_REPORT_INTERVAL)
        try:
            post_req = self.status_counter.post_req
            get_req = self.status_counter.get_req
            failed_req = self.status_counter.failed_req
            finish_req = self.status_counter.finish_req
            packed = struct.pack("<4I", post_req, get_req, failed_req, finish_req)
            message_buf[4 : 4 + len(packed)] = packed
        except Exception:
            pass

    @abstractmethod
    async def do_request(
        self, data: Any, token_bucket: BoundedSemaphore, session: aiohttp.ClientSession
    ) -> Any:
        """Call model to do request, return output. Must be async in your implementation.
        
        Args:
            data: Request data
            token_bucket: Semaphore for rate limiting
            session: HTTP session for the request
            
        Returns:
            Model output
            
        Raises:
            NotImplementedError: If not implemented in subclass
        """
        raise NotImplementedError

    def _read_and_unpickle(
        self, buf: memoryview, index_data: Tuple[int, int, int]
    ) -> Any:
        """Read and unpickle data from shared memory.
        
        Args:
            buf: Memory buffer
            index_data: Tuple of (index, offset, length)
            
        Returns:
            Unpickled data object
        """
        _, offset, length = index_data
        raw_bytes = buf[offset : offset + length]
        data_bytes = bytes(raw_bytes)
        return pickle.loads(data_bytes)

    def _get_single_data(
        self,
        share_memory: shared_memory.SharedMemory,
        index_queue: Queue,
    ) -> Optional[Any]:
        """Attempt to consume one token (if configured) and one index entry.

        All blocking operations are dispatched to a thread via asyncio.to_thread so the
        event loop is never blocked.
        Returns the deserialized data or None if there's no data / should stop.
        
        Args:
            share_memory: Shared memory containing data
            index_queue: Queue containing index information
            
        Returns:
            Deserialized data or None if no data available
        """

        try:
            # If unable to get index_data, exit directly
            index_data = index_queue.get(True, 1)
        except Exception as e:
            return None

        # Handle poison pill based on pressure_mode
        # Pressure mode: ignore poison pill, continue getting index_data (ensure other processes get data and immediately put back one)
        # Non-pressure mode: data exhausted, put new poison pill for other processes to exit, and return None
        if index_data is None:
            if self.pressure_mode:
                index_data = index_queue.get(True)
            else:
                # Put poison pill for other processes to exit
                index_queue.put(None, False)
                return None
        else:
            # In pressure mode, put one data back after getting one
            if self.pressure_mode:
                index_queue.put(index_data, False)
            # Return parsed data for model inference
            return self._read_and_unpickle(share_memory.buf, index_data)

    def _fill_janus_queue(
        self,
        dataset_share_memory: shared_memory.SharedMemory,
        index_queue: Queue,
        janus_queue: janus.Queue,
        stop_event: threading.Event,
    ):
        """Pre-fill janus queue with initial data.
        
        Args:
            dataset_share_memory: Shared memory containing dataset
            index_queue: Queue containing data indices
            janus_queue: Janus queue for thread-async communication
            stop_event: Event to signal termination
        """
        # Pre-fill up to batch_size items first (mirrors your original behavior)
        for _ in range(self.batch_size):
            if stop_event.is_set():
                break
            data = self._get_single_data(dataset_share_memory, index_queue)
            # Block if queue is full -> natural backpressure
            janus_queue.sync_q.put(data)
            if data is None:
                break

    def _producer_thread_target(
        self,
        dataset_share_memory: shared_memory.SharedMemory,
        index_queue: Queue,
        janus_queue: janus.Queue,
        stop_event: threading.Event,
    ) -> None:
        """Thread target: read from shared memory/index queue and push into janus.sync_q.
        
        Args:
            dataset_share_memory: Shared memory containing dataset
            index_queue: Queue containing data indices
            janus_queue: Janus queue for thread-async communication
            stop_event: Event to signal termination
        """
        logger.info("Producer thread started")
        try:
            # Continuous fill until stop_event or sentinel
            while not stop_event.is_set():
                data = self._get_single_data(dataset_share_memory, index_queue)
                janus_queue.sync_q.put(data)
                if data is None:
                    break
            logger.info("Producer thread finished")
        except Exception:
            # If producer errors, try to put sentinel so consumers can exit
            try:
                janus_queue.sync_q.put(None)
            except Exception:
                pass
            raise

    def _sync_main_process_with_message(
        self, message_share_memory: shared_memory.SharedMemory, info: int
    ):
        """Synchronize with main process using shared memory message.
        
        Args:
            message_share_memory: Shared memory for communication
            info: Information to send to main process
        """
        message_buf = message_share_memory.buf
        struct.pack_into("I", message_buf, 0, info)

    async def _worker_loop(
        self,
        token_bucket: BoundedSemaphore,
        async_queue: janus.Queue.async_q,
    ) -> None:
        """Worker task: repeatedly fetch data and call the async do_request.

        Consumes from janus.async_q (async_queue).
        
        Args:
            token_bucket: Semaphore for rate limiting
            async_queue: Async queue for data consumption
        """
        num_workers = self.batch_size if self.batch_size and self.batch_size > 0 else 1

        # Limit maximum concurrency
        semaphore = asyncio.Semaphore(num_workers) if num_workers else None
        # Reuse session to improve concurrency
        connector = aiohttp.TCPConnector(limit=self.batch_size + 1)
        timeout = aiohttp.ClientTimeout(total=REQUEST_TIME_OUT)
        session = aiohttp.ClientSession(
            connector=connector, timeout=timeout, max_line_size=MAX_CHUNK_SIZE
        )

        async def limited_request_func(data):
            if semaphore is None:
                return await self.do_request(data, token_bucket, session)
            async with semaphore:
                return await self.do_request(data, token_bucket, session)

        tasks = []
        try:
            while True:
                if token_bucket:
                    await asyncio.to_thread(token_bucket.acquire)
                    # await asyncio.sleep(0.01)
                else:
                    # Slightly limit RR when no token to avoid high CPU usage causing TTFT accumulation
                    await asyncio.sleep(BLOCK_INTERVAL)

                data = await async_queue.get()

                # data == None -> sentinel
                if data is None:
                    await async_queue.put(None)
                    break
                # Call user-provided async request
                tasks.append(asyncio.create_task(limited_request_func(data)))

            await asyncio.gather(*tasks)
        except Exception as e:
            for t in tasks:
                if not t.done():
                    t.cancel()
            # Wait for all tasks to finish, prevent "Task was destroyed but it is pending"
            await asyncio.gather(*tasks, return_exceptions=True)
            raise e
        finally:
            await session.close()

    def inference_with_shm(
        self,
        dataset_shm_name: str,
        message_shm_name: str,
        index_queue: Queue,
        token_bucket: BoundedSemaphore,
        output_json_filepath: Optional[str] = None,
    ) -> Dict[str, int]:
        """Top-level runner using janus for thread<->async bridging.

        This function is synchronous: it creates a dedicated asyncio loop, starts
        producer threads (which put into janus.sync_q) and then runs consumer
        tasks on the new loop which consume from janus.async_q.
        
        Args:
            dataset_shm_name: Name of dataset shared memory
            message_shm_name: Name of message shared memory
            index_queue: Queue containing data indices
            token_bucket: Semaphore for rate limiting
            output_json_filepath: Optional output file path
            
        Returns:
            Dictionary with status information
        """
        dataset_share_memory = shared_memory.SharedMemory(dataset_shm_name)
        message_share_memory = shared_memory.SharedMemory(message_shm_name)

        # status control
        stop_event = threading.Event()
        self.status_counter = StatusCounter(self.batch_size)
        self.status_counter.start()

        # create janus queue bound to that loop
        janus_queue = janus.Queue(maxsize=self.batch_size + 1)
        # start report thread
        report_thread = threading.Thread(
            target=self._monitor_status_thread,
            args=(stop_event, message_share_memory),
            daemon=True,
        )
        report_thread.start()

        self._fill_janus_queue(
            dataset_share_memory,
            index_queue,
            janus_queue,
            stop_event,
        )

        # start producer thread (fills janus_queue.sync_q)
        producer_thread = threading.Thread(
            target=self._producer_thread_target,
            args=(
                dataset_share_memory,
                index_queue,
                janus_queue,
                stop_event,
            ),
            daemon=True,
        )
        producer_thread.start()

        # Start cache consumer thread (preserve original behaviour)
        out_path = self.get_output_dir(output_json_filepath)
        tmp_json_filepath = os.path.join(out_path, "tmp")
        os.makedirs(tmp_json_filepath, exist_ok=True)
        tmp_file_name = f"tmp_{uuid.uuid4().hex[:8]}.jsonl"
        cache_consumer_thread = threading.Thread(
            target=self.output_handler.run_cache_consumer,
            args=(
                tmp_json_filepath,
                tmp_file_name,
                self.perf_mode,
                stop_event,
                self.save_every,
            ),
        )
        cache_consumer_thread.start()
        # Notify main process to start generating tokens
        self._sync_main_process_with_message(message_share_memory, 0)
        # Create a fresh event loop dedicated for running the async consumers
        loop = asyncio.new_event_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=self.batch_size)
        )
        asyncio.set_event_loop(loop)
        worker_task = loop.create_task(
            self._worker_loop(token_bucket, janus_queue.async_q)
        )
        # Run consumers on the created loop
        try:
            loop.run_until_complete(worker_task)
        except KeyboardInterrupt:
            stop_event.set()
        finally:
            # Orderly shutdown
            stop_event.set()

            janus_queue.sync_q.put(None)

            loop.run_until_complete(asyncio.wait_for(worker_task, timeout=10.0))
            # Close janus queue properly

            # Join threads
            cache_consumer_thread.join()
            producer_thread.join()
            report_thread.join()

            self.status_counter.stop()
            self.status_counter.join()

            janus_queue.close()
            loop.run_until_complete(janus_queue.wait_closed())

            loop.close()

            # Write data with same abbr to same jsonl file
            self.output_handler.write_to_json(out_path, self.perf_mode)

            dataset_share_memory.close()
            message_share_memory.close()

        return {"status": 0}


class StatusCounter(threading.Thread):
    """Thread-safe status counter for tracking request statistics."""
    
    def __init__(self, batch_size: int = 0):
        """Initialize status counter.
        
        Args:
            batch_size: Size of batch for queue capacity calculation
        """
        super().__init__(daemon=True)
        self.post_req = 0
        self.get_req = 0
        self.failed_req = 0
        self.finish_req = 0
        # Use thread-safe standard library queue with capacity equal to batch_size * 4
        if batch_size <= 0:
            self.status_queue = None
        self.status_queue: std_queue.Queue = std_queue.Queue(maxsize=batch_size * 4)
        self._stop_event = threading.Event()
        self._print_interval = 1.0  # Print status once per second

    # These maintain coroutine interface (caller uses await) but internally use synchronous put_nowait
    async def post(self):
        """Record a post request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("post_req")

    async def rev(self):
        """Record a get request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("get_req")

    async def failed(self):
        """Record a failed request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("failed_req")

    async def finish(self):
        """Record a finished request."""
        if not self.status_queue:
            return
        self.status_queue.put_nowait("finish_req")

    def stop(self):
        """Request thread to stop (called by main thread/coroutine)."""
        self._stop_event.set()

    def run(self):
        """
        Run in independent thread: print status at least once per second; continuously pull from queue and update counts.
        Use short polling with timeout=0.2 for better responsiveness, while using time accumulation to achieve once-per-second printing.
        """
        if not self.status_queue:
            return
        while not self._stop_event.is_set():
            try:
                # Small timeout to respond to stop requests promptly
                status = self.status_queue.get(timeout=0.2)
            except std_queue.Empty:
                status = None

            if status is not None:
                if status == "post_req":
                    self.post_req += 1
                elif status == "get_req":
                    self.get_req += 1
                elif status == "failed_req":
                    self.failed_req += 1
                elif status == "finish_req":
                    self.finish_req += 1

        # After stop request, try to consume remaining items in queue and update statistics (optional)
        while True:
            try:
                status = self.status_queue.get_nowait()
            except std_queue.Empty:
                break
            if status == "post_req":
                self.post_req += 1
            elif status == "get_req":
                self.get_req += 1
            elif status == "failed_req":
                self.failed_req += 1
            elif status == "finish_req":
                self.finish_req += 1
