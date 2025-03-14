import os
import csv
import json
import pytest
import shutil
from ais_bench.benchmark.datasets.custom import make_custom_dataset_config


ACC_EVALUATOR_NAME = "ais_bench.benchmark.openicl.icl_evaluator.icl_hf_evaluator.AccEvaluator"
CUSTOM_DATASET_NAME = "ais_bench.benchmark.datasets.custom.CustomDataset"
GEN_INFERENCER_NAME = "ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer.GenInferencer"
OPT_SIM_ACC_EVALUATOR_NAME = "ais_bench.benchmark.datasets.custom.OptionSimAccEvaluator"


class TestClass:

    @classmethod
    def setup_class(cls):
        cls.init(TestClass, cls)
    
    @classmethod
    def teardown_class(cls):
        if os.path.exists(cls.tmpdir):
            shutil.rmtree(cls.tmpdir)
            print(f"\n\n {cls.tmpdir} has been removed", end="")
            del cls.tmpdir
        print("\n ---class level teardown_class")

    def init(self, cls):
        curr_path = os.path.dirname(os.path.abspath(__file__))
        self.tmpdir = os.path.join(curr_path, "tmp_datasets")
        cls.tmpdir = self.tmpdir
        
        if os.path.exists(self.tmpdir):
            shutil.rmtree(self.tmpdir)
        os.makedirs(self.tmpdir)
        # 准备csv格式测试数据
        self.mcq_csv = os.path.join(self.tmpdir, "mcq_test.csv")
        self.qa_csv = os.path.join(self.tmpdir, "qa_test.csv")

        with open(self.mcq_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["question", "A", "B", "C", "answer"])
            writer.writerow(["1+1=", "2", "3", "4", "A"])

        with open(self.qa_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["question", "answer"])
            writer.writerow(["1+1=", "2"])

        # 准备QA测试数据
        self.mcq_jsonl = os.path.join(self.tmpdir, "mcq_test.jsonl")
        self.qa_jsonl = os.path.join(self.tmpdir, "qa_test.jsonl")
        
        with open(self.mcq_jsonl, 'w', encoding='utf-8') as f:
            json.dump({"question": "1+1=", "A": "2", "B": "3", "C": "4", "answer": "A"}, f)
            f.write('\n')

        with open(self.qa_jsonl, 'w', encoding='utf-8') as f:
            json.dump({"question": "1+1=", "answer": "2"}, f)
            f.write('\n')

    def test_mcq_csv_config_generation(self):
        """测试MCQ.csv类型配置生成"""
        config = {
            "path": str(self.mcq_csv)
        }
        
        result = make_custom_dataset_config(config)

        # 验证基础结构
        assert "abbr" in result
        assert result["type"] == CUSTOM_DATASET_NAME
        assert "mcq_test" in result["abbr"]
        assert result["path"] == str(self.mcq_csv)
        
        # 验证prompt模板
        reader_cfg = result["reader_cfg"]
        assert ['question', 'A', 'B', 'C'] == reader_cfg["input_columns"]
        assert "answer" == reader_cfg["output_column"]

        infer_cfg = result["infer_cfg"]
        assert "Question: {question}" in infer_cfg["prompt_template"]["template"]
        assert "A. {A}\nB. {B}\nC. {C}\n" in infer_cfg["prompt_template"]["template"]
        assert "Answer: {answer}" in infer_cfg["prompt_template"]["template"]
        
        # 验证评估器配置
        evaluator = result["eval_cfg"]["evaluator"]
        assert evaluator["type"] == OPT_SIM_ACC_EVALUATOR_NAME
        assert len(evaluator["options"]) == 3
        
        # 验证类型字符串化
        assert "CustomDataset" in result["type"]
        
    def test_mcq_jsonl_config_generation(self):
        """测试MCQ.jsonl类型配置生成"""
        config = {
            "path": str(self.mcq_jsonl)
        }
        
        result = make_custom_dataset_config(config)
        
        # 验证基础结构
        assert "abbr" in result
        assert result["type"] == CUSTOM_DATASET_NAME
        assert "mcq_test" in result["abbr"]
        assert result["path"] == str(self.mcq_jsonl)
        
        # 验证prompt模板
        reader_cfg = result["reader_cfg"]
        assert ['question', 'A', 'B', 'C'] == reader_cfg["input_columns"]
        assert "answer" == reader_cfg["output_column"]

        infer_cfg = result["infer_cfg"]
        assert "Question: {question}" in infer_cfg["prompt_template"]["template"]
        assert "A. {A}\nB. {B}\nC. {C}\n" in infer_cfg["prompt_template"]["template"]
        assert "Answer: {answer}" in infer_cfg["prompt_template"]["template"]
        
        # 验证评估器配置
        evaluator = result["eval_cfg"]["evaluator"]
        assert evaluator["type"] == OPT_SIM_ACC_EVALUATOR_NAME
        assert len(evaluator["options"]) == 3

        
        # 验证类型字符串化
        assert "CustomDataset" in result["type"]

    def test_qa_csv_config_generation(self):
        """测试QA.csv类型配置生成"""
        config = {
            "path": str(self.qa_csv)
        }
        
        result = make_custom_dataset_config(config)

        # 验证基础结构
        assert "abbr" in result
        assert result["type"] == CUSTOM_DATASET_NAME
        assert "qa_test" in result["abbr"]
        assert result["path"] == str(self.qa_csv)
        
        # 验证prompt模板
        reader_cfg = result["reader_cfg"]
        assert "question" in reader_cfg["input_columns"]
        assert "answer" == reader_cfg["output_column"]

        infer_cfg = result["infer_cfg"]
        assert "Question: {question}" in infer_cfg["prompt_template"]["template"]
        assert "Answer: {answer}" in infer_cfg["prompt_template"]["template"]
        
        # 验证评估器配置
        assert result["eval_cfg"]["evaluator"]["type"] == ACC_EVALUATOR_NAME
        
        # 验证类型字符串化
        assert "CustomDataset" in result["type"]
    
    def test_qa_jsonl_config_generation(self):
        """测试QA.jsonl类型配置生成"""
        config = {
            "path": str(self.qa_jsonl),
        }
        
        result = make_custom_dataset_config(config)

        # 验证prompt模板
        template = result["infer_cfg"]["prompt_template"]["template"]
        assert "Question: {question}" in template
        assert "Answer: {answer}" in template
        
        # 验证评估器类型
        assert result["eval_cfg"]["evaluator"]["type"] == ACC_EVALUATOR_NAME

    def test_custom_template_handling(self):
        """测试自定义模板处理"""
        custom_template = "Custom template: {question}\nChoices: {A}{B}{C}"
        config = {
            "path": str(self.mcq_csv),
            "template": custom_template
        }
        
        result = make_custom_dataset_config(config)
        assert custom_template in result["infer_cfg"]["prompt_template"]["template"]

    def test_edge_case_single_option(self):
        """测试边界情况：只有一个选项"""
        single_option = os.path.join(self.tmpdir, "single_option.csv")
        with open(single_option, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["question", "A", "answer"])
            writer.writerow(["Q?", "1", "A"])
        
        config = {"path": str(single_option)}
        result = make_custom_dataset_config(config)

        # 应识别为QA类型
        assert result["eval_cfg"]["evaluator"]["type"] == ACC_EVALUATOR_NAME