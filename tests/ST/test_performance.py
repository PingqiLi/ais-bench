import os
import json
import shutil
import sys
import logging
import pytest
from ais_bench.benchmark.cli.main import main

DATASETS_CONFIGS_LIST = [
    "mmlu",
    "gsm8k",
    "boolq",
    "ceval",
    "aime2024",
    "gpqa",
    "math",
    "synthetic",
]

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
        self._set_datasets_config_path(self)

    def _set_datasets_config_path(self):
        dataset_configs_base_dir = os.path.abspath(os.path.join(self.cur_dir, "../../ais_bench/benchmark/configs/datasets"))
        for dataset in DATASETS_CONFIGS_LIST:
            sys.path.append(os.path.join(dataset_configs_base_dir, dataset))

    def test_vllm_api_all_qwen2_7b_synthetic_0_shot(self, monkeypatch): 
        fake_prediction = "123321" 
        fake_time_str = "synthetic_0_shot" 
        datasets_abbr_name = "vllm-api-gpu-synthetic" 
        datasets_script_name = "synthetic_gen" 

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/synthetic.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/synthetic.json")
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



