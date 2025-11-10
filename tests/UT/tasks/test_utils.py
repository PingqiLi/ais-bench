import unittest
import struct
import time
from unittest.mock import patch, MagicMock
from multiprocessing import Event, shared_memory

import numpy as np
from mmengine.config import ConfigDict

from ais_bench.benchmark.tasks.utils import (
    create_message_share_memory,
    check_virtual_memory_usage,
    ProgressBar,
    TokenProducer,
    update_global_data_index,
    FMT,
    MESSAGE_SIZE,
    WAIT_FLAG,
    INDEX_READ_FLAG,
    MAX_VIRTUAL_MEMORY_USAGE_PERCENT,
    MESSAGE_INFO
)
from ais_bench.benchmark.tasks.base import TaskStateManager
from ais_bench.benchmark.utils.logging.error_codes import TINFER_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchRuntimeError, ParameterValueError


class TestCreateMessageShareMemory(unittest.TestCase):
    """测试create_message_share_memory函数"""

    def test_create_shared_memory(self):
        """测试创建共享内存"""
        shm = create_message_share_memory()
        
        self.assertIsNotNone(shm)
        self.assertEqual(shm.size, MESSAGE_SIZE)
        
        # 验证初始值
        status, post, recv, fail, finish, _, data_index = struct.unpack(FMT, shm.buf)
        self.assertEqual(status, 0)
        self.assertEqual(post, 0)
        self.assertEqual(recv, 0)
        self.assertEqual(fail, 0)
        self.assertEqual(finish, 0)
        self.assertEqual(data_index, INDEX_READ_FLAG)
        
        # 清理
        shm.close()
        shm.unlink()


class TestCheckVirtualMemoryUsage(unittest.TestCase):
    """测试check_virtual_memory_usage函数"""

    @patch('ais_bench.benchmark.tasks.utils.psutil.virtual_memory')
    @patch('ais_bench.benchmark.tasks.utils.logger')
    def test_memory_usage_within_threshold(self, mock_logger, mock_virtual_memory):
        """测试内存使用在阈值内的情况"""
        # 模拟内存使用率为50%
        mock_memory = MagicMock()
        mock_memory.total = 100 * 1024**3  # 100GB
        mock_memory.used = 40 * 1024**3  # 40GB
        mock_memory.available = 60 * 1024**3  # 60GB
        mock_virtual_memory.return_value = mock_memory
        
        dataset_bytes = 10 * 1024**3  # 10GB
        
        # 不应该抛出异常
        check_virtual_memory_usage(dataset_bytes)
        
        # 验证记录日志
        self.assertTrue(mock_logger.info.called)

    @patch('ais_bench.benchmark.tasks.utils.psutil.virtual_memory')
    def test_memory_usage_exceeds_threshold(self, mock_virtual_memory):
        """测试内存使用超过阈值的情况"""
        # 模拟内存使用率为85%
        mock_memory = MagicMock()
        mock_memory.total = 100 * 1024**3  # 100GB
        mock_memory.used = 75 * 1024**3  # 75GB
        mock_memory.available = 25 * 1024**3  # 25GB
        mock_virtual_memory.return_value = mock_memory
        
        dataset_bytes = 10 * 1024**3  # 10GB
        
        # 应该抛出异常
        with self.assertRaises(AISBenchRuntimeError) as context:
            check_virtual_memory_usage(dataset_bytes)
        
        error_code = context.exception.error_code_str
        self.assertEqual(error_code, TINFER_CODES.VIRTUAL_MEMORY_USAGE_TOO_HIGH.full_code)

    @patch('ais_bench.benchmark.tasks.utils.psutil.virtual_memory')
    @patch('ais_bench.benchmark.tasks.utils.logger')
    def test_custom_threshold(self, mock_logger, mock_virtual_memory):
        """测试自定义阈值"""
        mock_memory = MagicMock()
        mock_memory.total = 100 * 1024**3
        mock_memory.used = 50 * 1024**3
        mock_memory.available = 50 * 1024**3
        mock_virtual_memory.return_value = mock_memory
        
        dataset_bytes = 10 * 1024**3
        custom_threshold = 70
        
        # 使用自定义阈值，不应该抛出异常
        check_virtual_memory_usage(dataset_bytes, threshold_percent=custom_threshold)
        self.assertTrue(mock_logger.info.called)


class TestProgressBar(unittest.TestCase):
    """测试ProgressBar类"""

    def setUp(self):
        """设置测试环境"""
        self.stop_event = Event()
        self.per_pid_shms = {}
        self.data_num = 100
        self.finish_data_num = 0
        self.debug = False
        self.pressure = False

    def tearDown(self):
        """清理测试环境"""
        # 清理共享内存
        for shm in self.per_pid_shms.values():
            try:
                shm.close()
                shm.unlink()
            except:
                pass

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_init(self, mock_logger_class):
        """测试ProgressBar初始化"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        # 创建共享内存
        shm = create_message_share_memory()
        self.per_pid_shms[12345] = shm
        
        progress_bar = ProgressBar(
            per_pid_shms=self.per_pid_shms,
            stop_event=self.stop_event,
            data_num=self.data_num,
            finish_data_num=self.finish_data_num,
            debug=self.debug,
            pressure=self.pressure
        )
        
        self.assertEqual(progress_bar.data_num, self.data_num)
        self.assertEqual(progress_bar.finish_data_num, self.finish_data_num)
        self.assertEqual(progress_bar.total_data_num, self.data_num)
        self.assertEqual(progress_bar.debug, self.debug)
        self.assertEqual(progress_bar.pressure, self.pressure)
        self.assertEqual(progress_bar.stats, {"post": 0, "recv": 0, "fail": 0, "finish": 0})

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_recalc_aggregate(self, mock_logger_class):
        """测试_recalc_aggregate方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        # 设置per_pid_stats
        progress_bar.per_pid_stats = {
            1: {"post": 10, "recv": 8, "fail": 1, "finish": 7},
            2: {"post": 15, "recv": 12, "fail": 2, "finish": 10}
        }
        
        progress_bar._recalc_aggregate()
        
        self.assertEqual(progress_bar.stats["post"], 25)
        self.assertEqual(progress_bar.stats["recv"], 20)
        self.assertEqual(progress_bar.stats["fail"], 3)
        self.assertEqual(progress_bar.stats["finish"], 17)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_read_shared_memory_and_update_per_pid(self, mock_logger_class):
        """测试_read_shared_memory_and_update_per_pid方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        # 创建共享内存并设置值
        shm = create_message_share_memory()
        self.per_pid_shms[12345] = shm
        
        # 设置共享内存的值
        shm.buf[:] = struct.pack(FMT, 0, 10, 8, 1, 7, 0, INDEX_READ_FLAG)
        
        progress_bar = ProgressBar(
            per_pid_shms=self.per_pid_shms,
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        updated = progress_bar._read_shared_memory_and_update_per_pid()
        
        self.assertTrue(updated)
        self.assertIn(12345, progress_bar.per_pid_stats)
        self.assertEqual(progress_bar.per_pid_stats[12345]["post"], 10)
        self.assertEqual(progress_bar.stats["post"], 10)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_compute_rates_since_start(self, mock_logger_class):
        """测试_compute_rates_since_start方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.stats = {"post": 100, "recv": 80, "fail": 5, "finish": 75}
        
        # 等待一小段时间
        time.sleep(0.1)
        
        rates = progress_bar._compute_rates_since_start()
        
        self.assertIn("post", rates)
        self.assertIn("recv", rates)
        self.assertIn("fail", rates)
        self.assertIn("finish", rates)
        self.assertGreater(rates["post"], 0)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_compute_rates_interval(self, mock_logger_class):
        """测试_compute_rates_interval方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.stats = {"post": 100, "recv": 80, "fail": 5, "finish": 75}
        progress_bar._last_snapshot_stats = {"post": 50, "recv": 40, "fail": 2, "finish": 35}
        
        time.sleep(0.1)
        
        rates = progress_bar._compute_rates_interval()
        
        self.assertIn("post", rates)
        self.assertIn("recv", rates)
        self.assertIn("fail", rates)
        self.assertIn("finish", rates)

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_format_per_pid_brief(self, mock_logger_class):
        """测试_format_per_pid_brief方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.per_pid_stats = {
            12345: {"post": 10, "recv": 8, "fail": 1, "finish": 7},
            67890: {"post": 15, "recv": 12, "fail": 2, "finish": 10}
        }
        
        result = progress_bar._format_per_pid_brief()
        
        self.assertIn("12345", result)
        self.assertIn("67890", result)
        self.assertIn("7/10/8/1", result)  # finish/post/recv/fail

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_format_per_pid_brief_empty(self, mock_logger_class):
        """测试_format_per_pid_brief方法在没有worker时"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        progress_bar = ProgressBar(
            per_pid_shms={},
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        progress_bar.per_pid_stats = {}
        
        result = progress_bar._format_per_pid_brief()
        self.assertEqual(result, "<no workers>")

    @patch('ais_bench.benchmark.tasks.utils.AISLogger')
    def test_set_message_flag(self, mock_logger_class):
        """测试set_message_flag方法"""
        mock_logger = MagicMock()
        mock_logger_class.return_value = mock_logger
        
        shm = create_message_share_memory()
        self.per_pid_shms[12345] = shm
        
        progress_bar = ProgressBar(
            per_pid_shms=self.per_pid_shms,
            stop_event=self.stop_event,
            data_num=self.data_num
        )
        
        flag = 1
        progress_bar.set_message_flag(flag)
        
        # 验证标志被设置
        status, _, _, _, _, _, _ = struct.unpack(FMT, shm.buf)
        self.assertEqual(status, flag)
        
        # 验证记录日志
        mock_logger.debug.assert_called()


if __name__ == '__main__':
    unittest.main()

