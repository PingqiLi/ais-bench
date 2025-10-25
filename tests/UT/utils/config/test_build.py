import unittest
from unittest.mock import patch
from ais_bench.benchmark.utils.config import build_model_from_cfg,validate_model_cfg
class MockModels:
    @staticmethod
    def build(cfg):
        return {"model": "mock_model", "config": cfg}

class TestModelCfgValidation(unittest.TestCase):

    @patch("os.path.exists", return_value=True)
    def test_validate_model_cfg_valid(self, mock_exists):
        cfg = {
            "attr": "local",
            "abbr": "test-model",
            "path": "/some/valid/path",
            "model": "gpt-4",
            "request_rate": 100,
            "retry": 5,
            "host_ip": "127.0.0.1",
            "host_port": 8080,
            "max_out_len": 512,
            "batch_size": 32,
            "generation_kwargs": {},
            "type": "some.Type"
        }
        errors = validate_model_cfg(cfg)
        self.assertEqual(errors, {})

    @patch("os.path.exists", return_value=False)
    def test_validate_model_cfg_invalid_path(self, mock_exists):
        cfg = {"path": "/invalid/path"}
        errors = validate_model_cfg(cfg)
        self.assertIn("path", errors)

    def test_validate_model_cfg_invalid_ip(self):
        cfg = {"host_ip": "invalid_ip"}
        errors = validate_model_cfg(cfg)
        self.assertIn("host_ip", errors)

    def test_validate_model_cfg_invalid_abbr(self):
        cfg = {"abbr": 123}  # abbr 应该是字符串，但传入了数字
        errors = validate_model_cfg(cfg)
        self.assertIn("abbr", errors)

    def test_validate_model_cfg_invalid_batch_size(self):
        cfg = {"batch_size": -5}
        errors = validate_model_cfg(cfg)
        self.assertIn("batch_size", errors)

    @patch("os.path.exists", return_value=True)
    @patch("ais_bench.benchmark.utils.config.build.MODELS")
    def test_build_model_from_cfg_success(self, mock_models, mock_exists):
        mock_models.build.return_value = {"model": "mock_model", "config": {}}
        cfg = {
            "attr": "local",
            "abbr": "test-model",
            "path": "/some/valid/path",
            "model": "gpt-4",
            "request_rate": 100,
            "retry": 5,
            "host_ip": "127.0.0.1",
            "host_port": 8080,
            "max_out_len": 512,
            "batch_size": 32,
            "generation_kwargs": {},
            "type": "some.Type",
            "run_cfg": {},
            "summarizer_abbr": "summ",
            "pred_postprocessor": None,
            "min_out_len": 10
        }
        result = build_model_from_cfg(cfg)
        self.assertIn("model", result)
        self.assertEqual(result["model"], "mock_model")
        # 验证 MODELS.build 被调用
        mock_models.build.assert_called_once()

    @patch("os.path.exists", return_value=True)
    def test_build_model_from_cfg_failure(self, mock_exists):
        cfg = {
            "attr": "invalid",
            "type": "model.Type"
        }
        with self.assertRaises(ValueError) as context:
            build_model_from_cfg(cfg)
        self.assertIn("attr", str(context.exception))


if __name__ == "__main__":
    unittest.main()
