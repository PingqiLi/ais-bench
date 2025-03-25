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
    "mmlu_pro",
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

    # mode all
    def test_vllm_api_all_mmlu_5_shot_str(self, monkeypatch):
        from mmlu_gen_5_shot_str import mmlu_all_sets
        fake_prediction = "Answer: A"
        fake_time_str = "mmlu_gen_5_shot_str"
        datasets_abbr_name = "lukaemon_mmlu_"
        datasets_script_name = "mmlu_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in mmlu_all_sets:
            curr_datasets_abbr_name = datasets_abbr_name + category

            # check infer out
            infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{curr_datasets_abbr_name}.json")
            assert os.path.exists(infer_outputs_json_path)
            with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get(f"0").get("prediction") == fake_prediction

            # check eval out
            results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{curr_datasets_abbr_name}.json")
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

    def test_vllm_api_all_ceval_5_shot_str(self, monkeypatch):
        from ceval_gen_5_shot_str import ceval_all_sets
        fake_prediction = "A"
        fake_time_str = "ceval_gen_5_shot_str"
        datasets_abbr_name = "ceval-"
        datasets_script_name = "ceval_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in ceval_all_sets:
            curr_datasets_abbr_name = datasets_abbr_name + category

            # check infer out
            infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{curr_datasets_abbr_name}.json")
            assert os.path.exists(infer_outputs_json_path)
            with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get(f"0").get("prediction") == fake_prediction

            # check eval out
            results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{curr_datasets_abbr_name}.json")
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

    def test_vllm_api_all_ceval_0_shot_str(self, monkeypatch):
        from ceval_gen_0_shot_str import ceval_all_sets
        fake_prediction = "A"
        fake_time_str = "ceval_gen_0_shot_str"
        datasets_abbr_name = "ceval-"
        datasets_script_name = "ceval_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in ceval_all_sets:
            curr_datasets_abbr_name = datasets_abbr_name + category

            # check infer out
            infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{curr_datasets_abbr_name}.json")
            assert os.path.exists(infer_outputs_json_path)
            with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get(f"0").get("prediction") == fake_prediction

            # check eval out
            results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{curr_datasets_abbr_name}.json")
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

    def test_vllm_api_all_boolq_0_shot_str(self, monkeypatch):
        fake_prediction = "Yes"
        fake_time_str = "SuperGLUE_BoolQ_gen_0_shot_str"
        datasets_abbr_name = "BoolQ"
        datasets_script_name = "SuperGLUE_BoolQ_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_aime2024_0_shot_str(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "aime2024_gen_0_shot_str"
        datasets_abbr_name = "aime2024"
        datasets_script_name = "aime2024_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_gpqa_0_shot_str(self, monkeypatch):
        fake_prediction = "A"
        fake_time_str = "gpqa_gen_0_shot_str"
        datasets_abbr_name = "GPQA_diamond"
        datasets_script_name = "gpqa_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_math500_0_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "math500_0_shot"
        datasets_abbr_name = "math_prm800k_500"
        datasets_script_name = "math_prm800k_500_0shot_cot_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_math500_5_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "math500_5_shot"
        datasets_abbr_name = "math_prm800k_500"
        datasets_script_name = "math_prm800k_500_5shot_cot_gen"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_drop_0_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "drop_gen_0_shot_str"
        datasets_abbr_name = "drop"
        datasets_script_name = "drop_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_drop_3_shot(self, monkeypatch):
        fake_prediction = "11"
        fake_time_str = "drop_gen_3_shot_str"
        datasets_abbr_name = "drop"
        datasets_script_name = "drop_gen_3_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_humaneval_0_shot(self, monkeypatch):
        fake_prediction = "xxxx"
        fake_time_str = "humaneval_0_shot"
        datasets_abbr_name = "openai_humaneval"
        datasets_script_name = "humaneval_gen_0_shot"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{datasets_abbr_name}.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("humaneval_pass@1") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)

    def test_vllm_api_all_qwen2_7b_mmlu_pro_0_shot(self, monkeypatch):
        from mmlu_pro_categories import categories
        fake_prediction = "Answer: A"
        fake_time_str = "mmlu_pro_gen_0_shot"
        datasets_abbr_name = "mmlu_pro_"
        datasets_script_name = "mmlu_pro_gen_0_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in categories:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

            # check infer out
            infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{curr_datasets_abbr_name}.json")
            assert os.path.exists(infer_outputs_json_path)
            with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get(f"0").get("prediction") == fake_prediction

            # check eval out
            results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{curr_datasets_abbr_name}.json")
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

    def test_vllm_api_all_qwen2_7b_mmlu_pro_5_shot(self, monkeypatch):
        from mmlu_pro_categories import categories
        fake_prediction = "Answer: A"
        fake_time_str = "mmlu_pro_gen_5_shot_str"
        datasets_abbr_name = "mmlu_pro_"
        datasets_script_name = "mmlu_pro_gen_5_shot_str"

        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general", "--datasets", datasets_script_name,
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api.VLLMCustomAPI._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        for category in categories:
            curr_datasets_abbr_name = datasets_abbr_name + category.replace(" ", "_")

            # check infer out
            infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general/{curr_datasets_abbr_name}.json")
            assert os.path.exists(infer_outputs_json_path)
            with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            assert data.get(f"0").get("prediction") == fake_prediction

            # check eval out
            results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general/{curr_datasets_abbr_name}.json")
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

    def test_vllm_api_general_chat_lcb_code_gen_lite_0_shot(self, monkeypatch):
        fake_prediction = "xxxxxxxxxxx"
        fake_time_str = "lcb_code_gen_lite_0_shot"
        datasets_abbr_name = "lcb_code_generation"
        datasets_script_name = "livecodebench_code_generate_lite_gen_0_shot_chat"
        monkeypatch.setattr('sys.argv',
            ["ais_bench", "--models", "vllm_api_general_chat", "--datasets", datasets_script_name, "--summarizer", "example",
            "--mode", "all", "-w", self.test_data_path])
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._get_service_model_path", lambda *arg: "qwen2")
        monkeypatch.setattr("ais_bench.benchmark.models.vllm_custom_api_chat.VLLMCustomAPIChat._generate", lambda *arg: fake_prediction)
        monkeypatch.setattr("ais_bench.benchmark.cli.main.get_current_time_str", lambda *arg: fake_time_str)
        main()

        # check infer out
        infer_outputs_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/predictions/vllm-api-general-chat/{datasets_abbr_name}.json")
        assert os.path.exists(infer_outputs_json_path)
        with open(infer_outputs_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get(f"0").get("prediction") == fake_prediction

        # check eval out
        results_json_path = os.path.join(self.test_data_path, f"{fake_time_str}/results/vllm-api-general-chat/{datasets_abbr_name}.json")
        with open(results_json_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        assert data.get("pass@1") is not None

        # check vis
        vis_csv_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.csv")
        assert os.path.exists(vis_csv_path)
        vis_txt_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.txt")
        assert os.path.exists(vis_txt_path)
        vis_md_path = os.path.join(self.test_data_path, f"{fake_time_str}/summary/summary_{fake_time_str}.md")
        assert os.path.exists(vis_md_path)