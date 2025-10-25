import unittest
from unittest.mock import patch
from ais_bench.benchmark.cli.utils import fill_model_path_if_datasets_need, get_config_type
from ais_bench.benchmark.utils.logging.exceptions import ConfigError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES


class TestUtils(unittest.TestCase):
    def test_fill_model_path_if_synthetic_with_tokenid(self):
        """测试当数据集是SyntheticDataset且Type为tokenid时，成功添加model_path"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset",
            "config": {"Type": "tokenid"}
        }

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证结果
        self.assertEqual(dataset_cfg.get("model_path"), "/path/to/model")

    def test_fill_model_path_if_synthetic_missing_model_path(self):
        """测试当数据集是SyntheticDataset且Type为tokenid但缺少model_path时，抛出ConfigError"""
        # 准备数据
        model_cfg = {}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset",
            "config": {"Type": "tokenid"}
        }

        # 验证异常
        with self.assertRaises(ConfigError) as context:
            fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证错误信息
        self.assertIn(UTILS_CODES.SYNTHETIC_DS_MISS_REQUIRED_PARAM, str(context.exception))
        self.assertIn("[path] in model config is required", str(context.exception))

    def test_fill_model_path_if_synthetic_not_synthetic_dataset(self):
        """测试当数据集不是SyntheticDataset时，不做任何操作"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.custom.CustomDataset",
            "config": {"Type": "tokenid"}
        }
        original_dataset_cfg = dataset_cfg.copy()

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证没有修改
        self.assertEqual(dataset_cfg, original_dataset_cfg)
        self.assertNotIn("model_path", dataset_cfg)

    def test_fill_model_path_if_synthetic_not_tokenid_type(self):
        """测试当数据集是SyntheticDataset但Type不是tokenid时，不做任何操作"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset",
            "config": {"Type": "string"}
        }
        original_dataset_cfg = dataset_cfg.copy()

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证没有修改
        self.assertEqual(dataset_cfg, original_dataset_cfg)
        self.assertNotIn("model_path", dataset_cfg)

    def test_fill_model_path_if_synthetic_missing_config(self):
        """测试当dataset_cfg缺少config字段时，不做任何操作"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset"
        }
        original_dataset_cfg = dataset_cfg.copy()

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证没有修改
        self.assertEqual(dataset_cfg, original_dataset_cfg)
        self.assertNotIn("model_path", dataset_cfg)

    def test_fill_model_path_if_synthetic_missing_type_in_config(self):
        """测试当dataset_cfg的config缺少Type字段时，不做任何操作"""
        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": "ais_bench.benchmark.datasets.synthetic.SyntheticDataset",
            "config": {}
        }
        original_dataset_cfg = dataset_cfg.copy()

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证没有修改
        self.assertEqual(dataset_cfg, original_dataset_cfg)
        self.assertNotIn("model_path", dataset_cfg)

    @patch('ais_bench.benchmark.cli.utils.get_config_type')
    def test_fill_model_path_if_synthetic_with_class_object(self, mock_get_config_type):
        """测试当dataset_cfg的type是类对象而不是字符串时的情况"""
        # 模拟get_config_type返回值
        mock_get_config_type.return_value = "ais_bench.benchmark.datasets.synthetic.SyntheticDataset"

        # 准备数据
        model_cfg = {"path": "/path/to/model"}
        dataset_cfg = {
            "type": object(),  # 模拟类对象
            "config": {"Type": "tokenid"}
        }

        # 调用函数
        fill_model_path_if_datasets_need(model_cfg, dataset_cfg)

        # 验证结果
        self.assertEqual(dataset_cfg.get("model_path"), "/path/to/model")
        mock_get_config_type.assert_called_once_with(dataset_cfg.get("type"))

    def test_get_config_type(self):
        """测试get_config_type函数"""
        # 测试字符串类型
        self.assertEqual(get_config_type("test_string"), "test_string")

        # 测试类类型（而不是实例），因为函数直接访问__name__属性
        class TestClass:
            pass

        # 修正：使用类本身而不是实例来测试
        expected_type = f"{TestClass.__module__}.{TestClass.__name__}"
        self.assertEqual(get_config_type(TestClass), expected_type)


if __name__ == '__main__':
    unittest.main()