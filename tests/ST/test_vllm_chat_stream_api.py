import os
import json
import shutil
import sys
import logging
import pytest
import requests
from unittest.mock import patch
from ais_bench.benchmark.cli.main import main

GSK8K_DATA_COUNT = 1319

class Response:
    def __init__(self):
        self.response = []
        for s in ["A","is","ben", "ch", "20"]:
            data = {
                'id': 'chatcmpl-edb1d22693f04ff08f25db773b48f44d',
                'object': 'chat.completion.chunk',
                'created': 1743234544,
                'model': '/data/weight/DeepSeek-R1-Distill-Qwen-7B/',
                'choices': [{'index': 0, 'delta': {'content': s}, 'logprobs': None, 'finish_reason': None}]
            }
            self.response.append(f"data: {json.dumps(data)}\n")
    def iter_content(self,*args):
        for content in self.response:
            yield content.encode()

class TestClass:
    @classmethod
    def setup_class(cls):
        """
        class level setup_class
        """
        cls.init(TestClass)

    @classmethod
    def teardown_class(cls):

        print('\n ---class level teardown_class')

    def init(self):
        self.cur_dir = os.path.dirname(os.path.abspath(__file__))
        self.test_data_path = os.path.abspath(os.path.join(self.cur_dir, "../testdatas"))
        if os.path.exists(self.test_data_path):
            shutil.rmtree(self.test_data_path)
        os.makedirs(self.test_data_path)

    #mode infer
    def test_mindie_stream_api_infer(self, monkeypatch):
        fake_prediction = "Aisbench20"
        fake_time_str = "fake_time_vllm_api_stream_chat"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_stream_chat", "--datasets", "gsm8k_gen",
            "--mode", "infer", "-w", self.test_data_path])
        monkeypatch.setattr(requests, 'post', lambda *args, **kwargs: Response())
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChatStream._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg, **kwargs: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-stream-chat/gsm8k.json")
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert os.path.exists(infer_outputs_json_path)
        assert len(data) == GSK8K_DATA_COUNT
        assert data.get(f"{GSK8K_DATA_COUNT - 1}").get("prediction") == fake_prediction
