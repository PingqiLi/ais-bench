import unittest
from unittest.mock import patch

from datasets import Dataset

from ais_bench.benchmark.datasets.base import BaseDataset


class DummyDataset(BaseDataset):
    @staticmethod
    def load(**kwargs):
        # 返回一个简单的Dataset
        return Dataset.from_list([
            {"text": "a"},
            {"text": "b"},
            {"text": "c"},
        ])


class TestBaseDataset(unittest.TestCase):
    def test_repeated_dataset_and_metadata(self):
        # n=2 确保重复采样，提供必需的reader_cfg参数
        ds = DummyDataset(
            reader_cfg={'input_columns': ['text'], 'output_column': None},
            k=1,
            n=2
        )
        # DatasetReader会将Dataset转换为DatasetDict，包含train和test
        # 验证是DatasetDict类型
        from datasets import DatasetDict
        self.assertIsInstance(ds.dataset, DatasetDict)
        # 验证每个split的长度加倍（原始3条 * 2 = 6条）
        self.assertEqual(len(ds.dataset['train']), 6)
        self.assertEqual(len(ds.dataset['test']), 6)
        # 验证添加的元数据字段存在
        first = ds.dataset['test'][0]
        self.assertIn("subdivision", first)
        self.assertIn("idx", first)


if __name__ == "__main__":
    unittest.main()
