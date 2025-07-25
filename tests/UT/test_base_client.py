import pytest
import threading
import json
import time
from typing import Optional
from ais_bench.benchmark.clients.base_client import (
    AisBenchClientException,
    raise_error,
    BaseClient,
    BaseStreamClient,
)
from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.utils.results import MiddleData
import urllib3.exceptions

# Dummy logger for testing
class DummyLogger:
    def __init__(self):
        self.errors = []
    def error(self, msg):
        self.errors.append(msg)

@pytest.fixture(autouse=True)
def patch_logger(monkeypatch):
    dummy = DummyLogger()
    monkeypatch.setattr('ais_bench.benchmark.clients.base_client.get_logger', lambda: dummy)
    return dummy


def test_AisBenchClientException_valid_message():
    msg = 'error occurred'
    ex = AisBenchClientException(msg)
    assert ex.get_message() == msg


def test_AisBenchClientException_empty_message():
    with pytest.raises(ValueError):
        AisBenchClientException("")


def test_AisBenchClientException_too_long_message():
    long_msg = 'x' * 512001
    with pytest.raises(ValueError):
        AisBenchClientException(long_msg)


def test_raise_error_increments_and_logs_and_raises(patch_logger):
    lock = threading.Lock()
    counter = {'failed_num': 0}
    with pytest.raises(AisBenchClientException) as excinfo:
        raise_error('fail', lock, counter)
    # counter incremented
    assert counter['failed_num'] == 1
    # logger recorded message
    assert 'fail' in patch_logger.errors[0]
    # exception message matches
    assert excinfo.value.get_message() == 'fail'

class DummyStreamClient(BaseStreamClient):
    def __init__(self):
        super().__init__(url='http://example.com', retry=0)
    
    def construct_request_body(self, inputs: str, parameters: Optional[dict] = None):
        return {'dummy': inputs}
    
    def process_stream_line(self, json_content):
        # simply return the json content as is
        return json_content

    # not used in this test
    def update_middle_data(self, res, inputs):
        inputs.output = json.dumps(res)


def test_preprocess_cur_line_normal(patch_logger):
    client = DummyStreamClient()
    line = 'normal line'
    assert client.preprocess_cur_line(line) == line


# Example of testing BaseClient.request with non-stream
class DummyClient(BaseClient):
    def __init__(self):
        super().__init__(url='http://example.com', retry=0)
        self._is_stream = False
    def construct_request_body(self, inputs: str, parameters: Optional[dict] = None):
        return {'input': inputs}
    def process_response(self, response, last_time_point):
        # simulate JSON response
        return {'out': response.data.decode()}
    def update_middle_data(self, res, inputs):
        inputs.output = res['out']


def test_request_success(monkeypatch):
    # prepare dummy response
    payload = {'res': 'ok'}
    class Resp:
        def __init__(self):
            self.data = json.dumps(payload).encode()
            self.status = 200
    dummy_resp = Resp()
    client = DummyClient()
    # patch do_request to return dummy_resp
    monkeypatch.setattr(DummyClient, 'do_request', lambda self, body, method: dummy_resp)
    inputs = MiddleData(input_data='data')
    res = client.request(inputs, parameters={})
    assert res == json.dumps(payload)


def test_request_error(monkeypatch, patch_logger):
    # simulate do_request raising AisBenchClientException
    client = DummyClient()
    
    def raise_client_exception(self, body, method):
        raise AisBenchClientException("Request timeout")
    
    monkeypatch.setattr(DummyClient, 'do_request', raise_client_exception)
    inputs = MiddleData(input_data='data')
    with pytest.raises(AisBenchClientException) as excinfo:
        client.request(inputs, parameters={})
    assert 'timeout' in excinfo.value.get_message().lower()
