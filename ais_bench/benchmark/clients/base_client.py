import json
import time
import re
from abc import abstractmethod, ABC

import urllib3
import threading

from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.utils.results import MiddleData
import urllib3.util
from http import HTTPStatus
from urllib3.exceptions import HTTPError

RETRY_ERROR_LIST = [104]

def _stream_data_split(stream_data_line):
    stream_data_line = stream_data_line.lstrip("data:").rstrip("\n\0")
    stream_data_line = stream_data_line.replace("}\x00{", "}{")
    stream_data_line = re.sub(r"\}\s*data:\s*{", "} ,{", stream_data_line)
    stream_data_line = "[" + stream_data_line + "]"
    json_obj_arr = json.loads(stream_data_line)
    return json_obj_arr


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
    with lock:
        request_counter['failed_num'] += 1
    logger = get_logger()
    logger.error(message)
    raise AisBenchClientException(message=message) from None

class BaseClient(ABC):
    def __init__(self, url, retry):
        self.logger = get_logger()
        self.valid_url = url
        self._timeout = None
        self._is_stream = False
        self.do_performance = False
        self.request_counter = dict(get_req_num=0, failed_num=0)
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

    def set_request_counter(self,  request_counter):
        self.request_counter = request_counter

    def set_performance(self):
        self.do_performance = True

    def close(self):
        self._http_pool_manager.clear()

    def rev_count(self):
        with self.lock:
            self.request_counter['get_req_num'] += 1

    def do_request(
        self,
        request_body: dict,
        request_method: str = "POST",
    ):
        return self._http_pool_manager.request(
            request_method,
            self.valid_url,
            headers={"Content-Type": "application/json"},
            body=json.dumps(request_body).encode(),
            timeout=self._timeout,
            preload_content=not self._is_stream,
        )

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
        inputs.chunk_time_point_list.append(start_time * 1000)
        try:
            response_raw = self.do_request(request_body, "POST")
        except HTTPError as e:
            raise_error(f"{e}. Please check your host_ip and host_port!", self.lock, self.request_counter)
        if response_raw.status != HTTPStatus.OK:
            raise_error(
                "Request failed, status is %r, response from server is %r." % (response_raw.status,
                                                                                response_raw.data.decode()),
                self.lock,
                self.request_counter
            )
        if not self._is_stream:
            try:
                res_ = self.process_response(response_raw, start_time)
            except json.JSONDecodeError:
                # 打印decode失败时的原始数据
                decode_data = response_raw.data.decode(errors="replace")
                raise_error(f"Failed to decode text JSON response. Raw data: {decode_data}", self.lock, self.request_counter)
            response = [self.update_middle_data(res_, inputs)]
        else:
            response = []
            try:
                for res_ in self.process_response(response_raw, start_time):
                    response.append(self.update_middle_data(res_, inputs))
            except ValueError as e:
                raise_error(f"Failed to process stream response. {e}", self.lock, self.request_counter)
        self.rev_count()
        self.update_request_time(inputs, start_time)
        return "".join(response)



class BaseStreamClient(BaseClient, ABC):
    def __init__(self, url, retry):
        super().__init__(url, retry)
        self._is_stream = True

    def preprocess_cur_line(self, cur_line: str) -> str:
        if cur_line == "engine callback timeout.":
            raise_error("Engine time out. The tokens generation might be incomplete.", self.lock, self.request_counter)
        return cur_line

    @abstractmethod
    def process_stream_line(self, json_content: dict) -> dict:
        pass

    def process_response(self, response, last_time_point):
        time_name = "prefill_time"
        for byte_line in response.stream():
            if byte_line == b"\n":
                continue
            cur_line = self.preprocess_cur_line(byte_line.decode())
            try:
                for json_content in _stream_data_split(cur_line):
                    cur_time_point = time.perf_counter()
                    response_dict = self.process_stream_line(json_content)
                    if time_name not in response_dict.keys():
                        response_dict[time_name] = (
                            cur_time_point - last_time_point
                        ) * 1000
                        response_dict["chunk_time_point"] = cur_time_point * 1000
                    yield response_dict
                    time_name = "decode_time"
                    last_time_point = time.perf_counter()
            except Exception as error:
                raise ValueError("[Error] %r! Response from server is: %r" % (error, cur_line))
