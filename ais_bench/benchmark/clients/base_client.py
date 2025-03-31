import json
import time
import re
from abc import abstractmethod, ABC

import urllib3

from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.utils.results import MiddleData


def _stream_data_split(stream_data_line):
    stream_data_line = stream_data_line.lstrip("data:").rstrip("\n\0")
    stream_data_line = stream_data_line.replace("}\x00{", "}{")
    stream_data_line = re.sub(r"\}\s*data:{", "} ,{", stream_data_line)
    stream_data_line = "[" + stream_data_line + "]"
    json_obj_arr = json.loads(stream_data_line)
    return json_obj_arr


class BaseClient(ABC):
    def __init__(self, url):
        self.logger = get_logger()
        self.valid_url = url
        self._timeout = None
        self._is_stream = False
        self.do_performance = False
        self._http_pool_manager = urllib3.PoolManager()

    def __del__(self):
        self.close()

    @abstractmethod
    def construct_request_body(
        self,
        inputs: MiddleData,
        parameters: dict = None,
    ) -> dict:
        pass

    @abstractmethod
    def process_response(self, response, last_time_point):
        pass

    @abstractmethod
    def update_middle_data(self, res, inputs):
        pass
    
    def update_request_time(self, input:MiddleData, start_time):
        if not self.do_performance:
            return
        input.start_time = start_time
        input.end_time = time.time()
        input.req_latency = (input.end_time - input.start_time) * 1000
        
    def set_performance(self):
        self.do_performance = True
        
    def close(self):
        self._http_pool_manager.clear()

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
        try:
            request_body = self.construct_request_body(
                inputs.input_data,
                parameters=parameters,
            )
            start_time = time.time()
            response_raw = self.do_request(request_body, "POST")
            if not self._is_stream:
                response_obj = json.loads(response_raw.data.decode())
                err_msg = response_obj.get("error", None)
                if err_msg:
                    self.logger.error(
                        "Request failed, response from server is {}.".format(err_msg)
                    )
                res_ = self.process_response(response_raw, start_time)
                response = [self.update_middle_data(res_, inputs)]
            else:
                response = []
                for res_ in self.process_response(response_raw, start_time):
                    response.append(self.update_middle_data(res_, inputs))
            self.update_request_time(inputs, start_time)

        except urllib3.exceptions.TimeoutError:
            self.logger.error("The http request timeout.")
        except urllib3.exceptions.RequestError as err:
            self.logger.error("Request failed and the reason is {}.".format(err))
        except json.JSONDecodeError as err:
            self.logger.error("Request failed and the reason is {}.".format(err))
        return "".join(response)


class BaseStreamClient(BaseClient, ABC):
    def __init__(self, url):
        super().__init__(url)
        self._is_stream = True

    def preprocess_cur_line(self, cur_line: str) -> str:
        if cur_line == "engine callback timeout.":
            self.logger(
                "[MIE02E000407] Engine time out. The tokens generation might be incomplete."
            )
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
                    cur_time_point = time.time()
                    response_dict = self.process_stream_line(json_content)
                    if time_name not in response_dict.keys():
                        response_dict[time_name] = (
                            cur_time_point - last_time_point
                        ) * 1000
                    yield response_dict
                    time_name = "decode_time"
                    last_time_point = time.time()
            except Exception as error:
                self.logger.error(
                    f"Request failed and the reason is {error}, response from server is: {cur_line}"
                )
