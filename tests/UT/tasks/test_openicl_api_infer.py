import unittest
import os
import tempfile
from unittest.mock import patch, MagicMock

from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.openicl_api_infer import OpenICLApiInferTask, run_single_inferencer
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.logging.error_codes import TINFER_CODES
from ais_bench.benchmark.utils.logging.exceptions import ParameterValueError


class TestRunSingleInferencer(unittest.TestCase):
    """测试run_single_inferencer函数"""

    @patch('ais_bench.benchmark.tasks.openicl_api_infer.ICL_INFERENCERS')
    def test_run_single_inferencer(self, mock_inferencers):
        """测试run_single_inferencer函数"""
        mock_inferencer = MagicMock()
        mock_inferencers.build.return_value = mock_inferencer
        
        model_cfg = ConfigDict({"type": "test_model"})
        inferencer_cfg = ConfigDict({"type": "test_inferencer"})
        shm_name = "test_shm"
        message_shm_name = "test_message_shm"
        max_concurrency = 10
        indexes = {0: (0, 0, 100)}
        
        from multiprocessing import BoundedSemaphore
        token_bucket = BoundedSemaphore(10)
        
        run_single_inferencer(
            model_cfg,
            inferencer_cfg,
            shm_name,
            message_shm_name,
            max_concurrency,
            indexes,
            token_bucket
        )
        
        mock_inferencers.build.assert_called_once()
        mock_inferencer.inference_with_shm.assert_called_once()


if __name__ == '__main__':
    unittest.main()

