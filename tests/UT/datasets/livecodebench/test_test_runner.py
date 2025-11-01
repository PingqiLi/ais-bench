import unittest
import sys
import os
import json
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    # test_runner使用相对导入from testing_util import run_test
    # 需要确保testing_util在路径中，然后导入test_runner
    # 由于test_runner使用相对导入，我们需要patch testing_util模块
    import ais_bench.benchmark.datasets.livecodebench.test_runner as test_runner_module
    test_runner = test_runner_module
    TEST_RUNNER_AVAILABLE = True
except ImportError:
    TEST_RUNNER_AVAILABLE = False
    test_runner = None


class TestRunnerTestBase(unittest.TestCase):
    """TestRunner测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not TEST_RUNNER_AVAILABLE:
            cls.skipTest(cls, "TestRunner modules not available")


class TestTestRunner(TestRunnerTestBase):
    """测试test_runner模块"""
    
    @patch('sys.stdin')
    @patch('sys.exit')
    @patch('ais_bench.benchmark.datasets.livecodebench.testing_util.run_test')
    @patch('builtins.print')
    def test_main_success(self, mock_print, mock_run_test, mock_exit, mock_stdin):
        """测试main函数成功执行"""
        # 模拟stdin输入 - test_runner使用json.load(sys.stdin)
        mock_stdin_json = {
            'sample': {'input_output': '{"inputs": ["test"], "outputs": ["result"]}'},
            'generation': 'def test(): return "result"',
            'debug': False,
            'timeout': 10
        }
        
        # test_runner中json.load会从sys.stdin读取，我们需要patch json.load
        with patch.object(test_runner_module, 'json') as mock_json:
            mock_json.load.return_value = mock_stdin_json
            mock_run_test.return_value = ([True], {})
            
            # 执行main函数
            test_runner_module.main()
            
            # 验证run_test被调用
            mock_run_test.assert_called_once_with(
                mock_stdin_json['sample'],
                test=mock_stdin_json['generation'],
                debug=False,
                timeout=10
            )
            # 验证打印了JSON输出
            self.assertTrue(mock_print.called)
            # 验证退出码为0
            mock_exit.assert_called_with(0)
    
    @patch('sys.exit')
    @patch('builtins.print')
    @patch('sys.stderr')
    def test_main_with_exception(self, mock_stderr, mock_print, mock_exit):
        """测试main函数异常情况"""
        # 模拟JSON加载失败
        with patch.object(test_runner_module, 'json') as mock_json:
            mock_json.load.side_effect = ValueError('Invalid JSON')
            # 执行main函数
            test_runner_module.main()
            
            # 验证打印了错误信息（两次：一次stdout，一次stderr）
            self.assertTrue(mock_print.called)
            # 验证退出码为2
            mock_exit.assert_called_with(2)
    
    @patch('sys.stdin')
    @patch('sys.exit')
    @patch('ais_bench.benchmark.datasets.livecodebench.testing_util.run_test')
    @patch('builtins.print')
    def test_main_with_default_params(self, mock_print, mock_run_test, mock_exit, mock_stdin):
        """测试main函数使用默认参数"""
        mock_stdin_json = {
            'sample': {'input_output': '{"inputs": ["test"], "outputs": ["result"]}'},
            'generation': 'def test(): return "result"'
            # 没有提供debug和timeout，应该使用默认值
        }
        
        with patch.object(test_runner_module, 'json') as mock_json:
            mock_json.load.return_value = mock_stdin_json
            mock_run_test.return_value = ([True], {})
            
            # 执行main函数
            test_runner_module.main()
            
            # 验证使用了默认值
            mock_run_test.assert_called_once_with(
                mock_stdin_json['sample'],
                test=mock_stdin_json['generation'],
                debug=False,
                timeout=10
            )
            mock_exit.assert_called_with(0)


if __name__ == '__main__':
    unittest.main()

