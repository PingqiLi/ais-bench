import unittest
from unittest.mock import patch, MagicMock

# 导入被测试的模块
from ais_bench.benchmark.utils.exceptions import (
    AISBenchBaseException,
    PerfResultCalcException
)


class TestAISBenchBaseException(unittest.TestCase):
    """测试AISBenchBaseException类"""

    def setUp(self):
        """设置测试环境"""
        # 模拟有效的错误码
        self.valid_error_str = "TEST-ERR-001"
        self.mock_error_code = MagicMock()
        self.mock_error_code.full_code = self.valid_error_str
        self.mock_error_code.message = "测试错误消息"

        # 模拟格式化后的日志内容
        self.formatted_log_content = "格式化后的日志内容"

    @patch('ais_bench.benchmark.utils.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.exceptions.get_formatted_log_content')
    def test_init_with_existing_error_code(self, mock_get_formatted, mock_manager):
        """测试使用存在的错误码初始化异常"""
        # 设置模拟对象的返回值
        mock_manager.get.return_value = self.mock_error_code
        mock_get_formatted.return_value = self.formatted_log_content

        # 创建异常实例
        exception = AISBenchBaseException(self.valid_error_str)

        # 验证调用了error_manager.get和get_formatted_log_content
        mock_manager.get.assert_called_once_with(self.valid_error_str)
        mock_get_formatted.assert_called_once_with(self.valid_error_str, None)

        # 验证异常的message属性
        self.assertEqual(str(exception), self.formatted_log_content)

    @patch('ais_bench.benchmark.utils.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.exceptions.get_formatted_log_content')
    def test_init_with_message(self, mock_get_formatted, mock_manager):
        """测试提供message参数的情况"""
        # 设置模拟对象的返回值
        mock_manager.get.return_value = self.mock_error_code
        mock_get_formatted.return_value = self.formatted_log_content

        # 创建异常实例并提供message参数
        custom_message = "自定义错误消息"
        exception = AISBenchBaseException(self.valid_error_str, custom_message)

        # 验证调用了error_manager.get和get_formatted_log_content
        mock_manager.get.assert_called_once_with(self.valid_error_str)
        mock_get_formatted.assert_called_once_with(self.valid_error_str, custom_message)

        # 验证异常的message属性
        self.assertEqual(str(exception), self.formatted_log_content)

    @patch('ais_bench.benchmark.utils.exceptions.error_manager')
    def test_init_with_nonexistent_error_code(self, mock_manager):
        """测试使用不存在的错误码初始化异常时抛出ValueError"""
        # 设置模拟对象返回None，表示错误码不存在
        nonexistent_error_str = "NONEXISTENT-ERR-999"
        mock_manager.get.return_value = None

        # 验证抛出ValueError
        with self.assertRaises(ValueError) as context:
            AISBenchBaseException(nonexistent_error_str)

        # 验证异常消息
        self.assertIn(f"error_code {nonexistent_error_str} is not exist!", str(context.exception))

        # 验证调用了error_manager.get
        mock_manager.get.assert_called_once_with(nonexistent_error_str)

    @patch('ais_bench.benchmark.utils.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.exceptions.get_formatted_log_content')
    def test_inheritance(self, mock_get_formatted, mock_manager):
        """测试异常类的继承关系"""
        # 设置模拟对象的返回值
        mock_manager.get.return_value = self.mock_error_code
        mock_get_formatted.return_value = self.formatted_log_content

        # 创建异常实例
        exception = AISBenchBaseException(self.valid_error_str)

        # 验证继承关系
        self.assertIsInstance(exception, Exception)
        self.assertIsInstance(exception, AISBenchBaseException)


class TestPerfResultCalcException(unittest.TestCase):
    """测试PerfResultCalcException类"""

    @patch('ais_bench.benchmark.utils.exceptions.error_manager')
    @patch('ais_bench.benchmark.utils.exceptions.get_formatted_log_content')
    def test_inheritance(self, mock_get_formatted, mock_manager):
        """测试PerfResultCalcException类的继承关系"""
        # 设置模拟对象的返回值
        mock_manager.get.return_value = MagicMock()
        mock_get_formatted.return_value = "格式化后的日志内容"

        # 创建异常实例
        exception = PerfResultCalcException("TEST-ERR-001")

        # 验证继承关系
        self.assertIsInstance(exception, Exception)
        self.assertIsInstance(exception, AISBenchBaseException)
        self.assertIsInstance(exception, PerfResultCalcException)


if __name__ == '__main__':
    unittest.main()