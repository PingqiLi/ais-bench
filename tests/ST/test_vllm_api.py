import os
import json
import shutil
import sys
import logging
import pytest
from ais_bench.benchmark.cli.main import main

GSK8K_DATA_COUNT = 1319


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

    # mode infer
    def test_vllm_api_general_infer_default(self, monkeypatch):
        fake_prediction = "test_response_for_vllm_api_general"
        fake_time_str = "fake_time_vllm_api_general_infer"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", "gsm8k_gen",
            "--mode", "infer", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/gsm8k.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert len(data) == GSK8K_DATA_COUNT
        assert data.get(f"{GSK8K_DATA_COUNT - 1}").get("prediction") == fake_prediction

    # mode all
    def test_vllm_api_general_all_default(self, monkeypatch):
        fake_prediction = "test_response_for_vllm_api_general"
        fake_time_str = "fake_time_vllm_api_general_all"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", "gsm8k_gen",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/gsm8k.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert len(data) == GSK8K_DATA_COUNT
        assert data.get(f"{GSK8K_DATA_COUNT - 1}").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/gsm8k.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("accuracy") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

