import unittest
from unittest.mock import patch, mock_open

from datasets import Dataset, DatasetDict

from ais_bench.benchmark.datasets.gpqa import (
    GPQADataset,
    GPQASimpleEvalDataset,
    GPQAEvaluator,
    GPQA_Simple_Eval_postprocess,
)
from ais_bench.benchmark.datasets.gsm8k import (
    GSM8KDataset,
    Gsm8kEvaluator,
    gsm8k_dataset_postprocess,
    gsm8k_postprocess,
)


class TestGPQA(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.gpqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_dataset(self, mock_open_file, mock_get_path):
        # CSV 头 + 一行数据；索引7是Question，8-11为选项
        content = (
            "h0,h1,h2,h3,h4,h5,h6,Question,A,B,C,D\n"
            ",,,,,,,Q,oa,ob,oc,od\n"
        )
        m = mock_open(read_data=content)
        mock_open_file.return_value = m.return_value
        ds = GPQADataset.load("/any", name="file.csv")
        self.assertIsInstance(ds, Dataset)
        self.assertEqual(len(ds), 1)

    @patch("ais_bench.benchmark.datasets.gpqa.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_simple_eval_dataset(self, mock_open_file, mock_get_path):
        content = (
            "h0,h1,h2,h3,h4,h5,h6,Question,A,B,C,D\n"
            ",,,,,,,Q,oa,ob,oc,od\n"
        )
        m = mock_open(read_data=content)
        mock_open_file.return_value = m.return_value
        ds = GPQASimpleEvalDataset.load("/any", name="file.csv")
        self.assertIsInstance(ds, Dataset)
        self.assertGreaterEqual(len(ds), 1)

    def test_evaluator_and_postprocess(self):
        eva = GPQAEvaluator()
        out = eva.score(["A"], ["A"])
        self.assertIn("accuracy", out)
        self.assertEqual(GPQA_Simple_Eval_postprocess("Answer: B"), "B")


class TestGSM8K(unittest.TestCase):
    @patch("ais_bench.benchmark.datasets.gsm8k.get_data_path", return_value="/fake/path")
    @patch("builtins.open")
    def test_dataset(self, mock_open_file, mock_get_path):
        line = '{"q": 1}'
        m = mock_open(read_data=line + "\n")
        # train 与 test
        mock_open_file.side_effect = [m.return_value, m.return_value]
        ds = GSM8KDataset.load("/any")
        self.assertIsInstance(ds, DatasetDict)
        self.assertIn("train", ds)
        self.assertIn("test", ds)

    def test_postprocess_and_evaluator(self):
        self.assertEqual(gsm8k_dataset_postprocess("x #### 1,234"), "1234")
        self.assertEqual(gsm8k_postprocess("12\nQuestion: 5"), "12")
        eva = Gsm8kEvaluator()
        out = eva.score(["5"], [5])
        self.assertIn("accuracy", out)


if __name__ == "__main__":
    unittest.main()
