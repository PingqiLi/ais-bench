import json
import time
import re
import queue
from abc import abstractmethod, ABC

import urllib3
import threading
import urllib3.util
import numpy as np
from http import HTTPStatus
from urllib3.exceptions import HTTPError, ReadTimeoutError, NewConnectionError, MaxRetryError

from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.utils.results import MiddleData
from ais_bench.benchmark.utils.valid_global_consts import valid_max_chunk_size, valid_request_time_out


RETRY_ERROR_LIST = [104]

class AisBenchClientException(Exception):
    def __init__(self, message):
        super().__init__()
        if len(message) == 0 or len(message) > 512000:
            raise ValueError("The length of message should be in range[1, 512000], \
                but got {}".format(len(message)))
        self._error_message = message

    def get_message(self):
        return self._error_message

def raise_error(message, lock, request_counter):
    request_counter.put_nowait("fail_req")
    logger = get_logger()
    logger.error(f"[AisBenchClientException] {message}")
    raise AisBenchClientException(message=message) from None

class BaseClient(ABC):
    def __init__(self, url, retry):
        self.logger = get_logger()
        self.valid_url = url
        self._timeout = valid_request_time_out()
        self.retry_num = retry
        self._is_stream = False
        self.do_performance = False
        self.request_counter = queue.SimpleQueue()
        self.lock = threading.Lock()
        retries = urllib3.util.Retry(
            total=retry,
            status_forcelist=RETRY_ERROR_LIST, # Retry if  Connection aborted
            allowed_methods=["POST"]
        )
        self._http_pool_manager = urllib3.PoolManager(retries=retries)

    def __del__(self):
        self.close()

    @abstractmethod
    def construct_request_body(
        self,
        inputs: MiddleData,
        parameters: dict = None,
    ) -> dict:
        pass

    def process_response(self, response, last_time_point):
        return json.loads(response.data.decode())

    @abstractmethod
    def update_middle_data(self, res, inputs):
        pass

    def update_request_time(self, input:MiddleData, start_time):
        if not self.do_performance:
            return
        input.start_time = start_time
        input.end_time = time.perf_counter()
        input.req_latency = (input.end_time - input.start_time) * 1000

    def set_performance(self):
        self.do_performance = True

    def close(self):
        self._http_pool_manager.clear()

    def rev_count(self):
        self.request_counter.put_nowait("get_req")
    
    def post_count(self):
        self.request_counter.put_nowait("post_req")
        
    def finish_count(self):
        self.request_counter.put_nowait("finish_req")

    def do_request(
        self,
        request_body: dict,
        request_method: str = "POST",
    ):
        try:
            self.post_count()
            raw_response = self._http_pool_manager.request(
            request_method,
            self.valid_url,
            headers={"Content-Type": "application/json","Authorization": "Bearer "+getattr(self, "api_key","")},
            body=json.dumps(request_body).encode(),
            timeout=self._timeout,
            preload_content=not self._is_stream,
        )
        except (ReadTimeoutError, TimeoutError) as e:
            raise_error(
                f"Request failed due to read timeout after {self._timeout}s.",
                self.lock,
                self.request_counter,
            )
        except NewConnectionError:
            raise_error(
                "Request failed: Unable to establish a new connection. Please verify your host IP and port.",
                self.lock,
                self.request_counter,
            )
        except MaxRetryError:
            raise_error(
                f"Request failed: Exceeded maximum retry attempts ({self.retry_num}).",
                self.lock,
                self.request_counter,
            )
        except HTTPError as e:
            raise_error(
                f"HTTP error occurred during request: {e}.",
                self.lock,
                self.request_counter,
            )
        except Exception as e:
            raise_error(
                f"Unexpected error during request: {e}.",
                self.lock,
                self.request_counter,
            )
        return raw_response

    def request(
        self,
        inputs: MiddleData,
        parameters: dict = None,
    ):
        response = None
        request_body = self.construct_request_body(
            inputs.input_data,
            parameters=parameters,
        )
        start_time = time.perf_counter()
        response_raw = self.do_request(request_body, "POST")
        if response_raw.status != HTTPStatus.OK:
            raise_error(
                f"Request failed: HTTP status {response_raw.status}. Server response: {response_raw.data.decode()}",
                self.lock,
                self.request_counter
            )
        if not self._is_stream:
            try:
                res_ = self.process_response(response_raw, start_time)
                self.update_middle_data(res_, inputs)
            except json.JSONDecodeError:
                decode_data = response_raw.data.decode(errors="replace")
                raise_error(f"Failed to decode JSON response. Raw data: {decode_data}", self.lock, self.request_counter)
            except RuntimeError as e:
                raise_error(f"{e}", self.lock, self.request_counter)
        else:
            try:
                for res_ in self.process_response(response_raw, start_time):
                    self.update_middle_data(res_, inputs)
            except ValueError as e:
                raise_error(f"Error processing stream response: {e}", self.lock, self.request_counter)
            except HTTPError as e:
                raise_error(f"HTTP error during stream response processing: {e}.", self.lock, self.request_counter)
            except Exception as e:
                raise_error(f"Other error during stream response processing: {e}.", self.lock, self.request_counter)
        self.rev_count()
        inputs.decode_cost = np.array(inputs.decode_cost, dtype=np.float32)
        self.update_request_time(inputs, start_time)
        return inputs.get_output()


class BaseStreamClient(BaseClient, ABC):
    def __init__(self, url, retry):
        super().__init__(url, retry)
        self._is_stream = True

    @abstractmethod
    def process_stream_line(self, json_content: dict) -> dict:
        pass

    def iter_lines(self, stream):
        """
        Split the input stream into lines based on multiple delimiters:
        - "\n\n" (LF LF)
        - "\r\n\r\n" (CRLF CRLF) 
        - "\r\r" (CR CR)
        
        If the received packet does not encounter any of these delimiters,
        cache it and concatenate it with the subsequent stream.
        """
        pending = None
        for chunk in stream:
            if pending is not None:
                chunk = pending + chunk
            lines = [d for d in chunk.replace(b'\r\n\r\n', b'\n\n').replace(b'\r\r', b'\n\n').split(b'\n\n') if d]
            # If there are no lines or the chunk is empty, clear pending
            if not lines or not chunk:
                pending = None
            # If the last line's last byte matches the chunk's last byte,
            # it means the chunk did not end with '\n\n', so the last segment is incomplete
            elif lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None
            yield from lines
        # After the stream ends, yield any remaining incomplete data
        if pending is not None:
            yield pending

    def process_response(self, response, last_time_point):
        time_name = "prefill_time"
        for raw_chunk in self.iter_lines(response.stream(amt=valid_max_chunk_size())):
            try:
                chunk = raw_chunk.decode().lstrip("data:").rstrip("\n\0").strip()
                if chunk == "[DONE]":
                    break
                chunk = json.loads(chunk)
                cur_time_point = time.perf_counter()
                response_dict = self.process_stream_line(chunk)
                if time_name not in response_dict.keys():
                    response_dict[time_name] = round((cur_time_point - last_time_point) * 1000, 4)
                yield response_dict
                time_name = "decode_time"
                last_time_point = time.perf_counter()
            except Exception as error:
                raise ValueError(f"[StreamResponseError] {error}! Raw server response: {raw_chunk}")
