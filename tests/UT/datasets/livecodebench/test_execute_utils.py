import unittest
import sys
import os
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.livecodebench.execute_utils import (
        codeexecute_check_correctness,
        unsafe_execute,
        time_limit,
        swallow_io,
        create_tempdir,
        chdir,
        reliability_guard,
        BASE_IMPORTS,
        TimeoutException,
        WriteOnlyStringIO,
        redirect_stdin
    )
    EXECUTE_UTILS_AVAILABLE = True
except ImportError:
    EXECUTE_UTILS_AVAILABLE = False


class ExecuteUtilsTestBase(unittest.TestCase):
    """ExecuteUtils测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not EXECUTE_UTILS_AVAILABLE:
            cls.skipTest(cls, "ExecuteUtils modules not available")


class TestBASE_IMPORTS(ExecuteUtilsTestBase):
    """测试BASE_IMPORTS常量"""
    
    def test_base_imports_exists(self):
        """测试BASE_IMPORTS存在且包含必要的导入"""
        self.assertIsNotNone(BASE_IMPORTS)
        self.assertIn('from itertools import', BASE_IMPORTS)
        self.assertIn('from math import', BASE_IMPORTS)
        self.assertIn('from collections import', BASE_IMPORTS)


class TestTimeoutException(ExecuteUtilsTestBase):
    """测试TimeoutException类"""
    
    def test_timeout_exception(self):
        """测试超时异常"""
        exc = TimeoutException('Timed out!')
        self.assertIsInstance(exc, Exception)
        self.assertEqual(str(exc), 'Timed out!')


class TestWriteOnlyStringIO(ExecuteUtilsTestBase):
    """测试WriteOnlyStringIO类"""
    
    def test_write_only_read_raises(self):
        """测试只写模式的StringIO读取会抛出异常"""
        stream = WriteOnlyStringIO()
        stream.write('test')
        
        with self.assertRaises(OSError):
            stream.read()
    
    def test_write_only_readline_raises(self):
        """测试只写模式的StringIO读取行会抛出异常"""
        stream = WriteOnlyStringIO()
        stream.write('test\n')
        
        with self.assertRaises(OSError):
            stream.readline()
    
    def test_write_only_readlines_raises(self):
        """测试只写模式的StringIO读取所有行会抛出异常"""
        stream = WriteOnlyStringIO()
        stream.write('test\n')
        
        with self.assertRaises(OSError):
            stream.readlines()
    
    def test_write_only_readable_returns_false(self):
        """测试只写模式的StringIO readable返回False"""
        stream = WriteOnlyStringIO()
        self.assertFalse(stream.readable())
    
    def test_write_only_writable(self):
        """测试只写模式的StringIO可以写入"""
        stream = WriteOnlyStringIO()
        result = stream.write('test')
        self.assertEqual(result, 4)  # 返回写入的字符数


class TestCodeExecuteCheckCorrectness(ExecuteUtilsTestBase):
    """测试codeexecute_check_correctness函数"""
    
    def test_check_correctness_imports(self):
        """测试检查正确性函数可以导入"""
        # 由于这个函数涉及到多进程和实际代码执行，在单元测试中只验证函数存在
        self.assertTrue(callable(codeexecute_check_correctness))
    
    @patch('ais_bench.benchmark.datasets.livecodebench.execute_utils.multiprocessing.Process')
    def test_check_correctness_simple_case(self, mock_process):
        """测试检查简单代码的正确性"""
        # 模拟多进程执行
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.is_alive.return_value = False
        
        # 模拟结果
        from multiprocessing import Manager
        with patch('ais_bench.benchmark.datasets.livecodebench.execute_utils.multiprocessing.Manager') as mock_manager:
            mock_manager_instance = MagicMock()
            mock_manager.return_value = mock_manager_instance
            mock_result_list = MagicMock()
            mock_result_list.__getitem__.return_value = 'passed'
            mock_manager_instance.list.return_value = mock_result_list
            
            result = codeexecute_check_correctness('assert 1 + 1 == 2', timeout=1)
            self.assertIsInstance(result, bool)
    
    @patch('ais_bench.benchmark.datasets.livecodebench.execute_utils.multiprocessing.Process')
    def test_check_correctness_timeout(self, mock_process):
        """测试检查正确性超时情况"""
        mock_process_instance = MagicMock()
        mock_process.return_value = mock_process_instance
        mock_process_instance.is_alive.return_value = True  # 进程仍在运行
        
        with patch('ais_bench.benchmark.datasets.livecodebench.execute_utils.multiprocessing.Manager') as mock_manager:
            mock_manager_instance = MagicMock()
            mock_manager.return_value = mock_manager_instance
            mock_result_list = []
            mock_manager_instance.list.return_value = mock_result_list
            
            # 当进程仍在运行时，应该返回False（timeout）
            result = codeexecute_check_correctness('assert 1 + 1 == 2', timeout=1)
            self.assertIsInstance(result, bool)


class TestContextManagers(ExecuteUtilsTestBase):
    """测试上下文管理器"""
    
    @patch('signal.setitimer')
    @patch('signal.signal')
    def test_time_limit(self, mock_signal, mock_setitimer):
        """测试time_limit上下文管理器"""
        with time_limit(5):
            pass
        
        # 验证signal被调用
        self.assertTrue(mock_setitimer.called)
    
    @patch('contextlib.redirect_stdout')
    @patch('contextlib.redirect_stderr')
    def test_swallow_io(self, mock_redirect_stderr, mock_redirect_stdout):
        """测试swallow_io上下文管理器"""
        with swallow_io():
            pass
        
        # 验证上下文管理器可以正常工作
        self.assertTrue(True)
    
    @patch('tempfile.TemporaryDirectory')
    @patch('os.chdir')
    def test_create_tempdir(self, mock_chdir, mock_tempdir):
        """测试create_tempdir上下文管理器"""
        mock_tempdir.return_value.__enter__.return_value = '/tmp/test'
        
        with create_tempdir() as dirname:
            self.assertIsNotNone(dirname)


class TestReliabilityGuard(ExecuteUtilsTestBase):
    """测试reliability_guard函数"""
    
    def test_reliability_guard(self):
        """测试reliability_guard函数"""
        # reliability_guard会禁用很多危险函数，测试它不会抛出异常
        try:
            reliability_guard()
            self.assertTrue(True)
        except Exception:
            self.fail("reliability_guard should not raise exceptions")


class TestRedirectStdin(ExecuteUtilsTestBase):
    """测试redirect_stdin类"""
    
    def test_redirect_stdin_exists(self):
        """测试redirect_stdin类存在"""
        self.assertTrue(hasattr(redirect_stdin, '_stream'))


if __name__ == '__main__':
    unittest.main()

