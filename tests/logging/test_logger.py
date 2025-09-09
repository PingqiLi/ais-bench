import unittest
from unittest.mock import patch, MagicMock, mock_open
import logging
import sys
from io import StringIO
import time

# 导入被测试的模块
from ais_bench.benchmark.utils.logger import (
    Colors,
    ColoredLevelFormatter,
    LOG_NORMAL_FORMATTER,
    LOG_DEBUG_FORMATTER,
    to_error_code_format,
    to_url_format,
    to_code_msg_format,
    get_formatted_log_content,
    AISLogger
)
from ais_bench.benchmark.utils.error_codes import ErrorType


class TestFormattingFunctions(unittest.TestCase):
    """测试格式化相关的工具函数"""

    def test_to_error_code_format(self):
        """测试错误码格式化函数"""
        msg = "TEST-CODE"
        expected = f"{Colors.BOLD}{Colors.BG_RED}{Colors.YELLOW}{msg}{Colors.RESET}"
        result = to_error_code_format(msg)
        self.assertEqual(result, expected)

    def test_to_url_format(self):
        """测试URL格式化函数"""
        msg = "https://example.com"
        expected = f"{Colors.UNDERLINE}{Colors.MAGENTA}{msg}{Colors.RESET}"
        result = to_url_format(msg)
        self.assertEqual(result, expected)

    def test_to_code_msg_format(self):
        """测试代码消息格式化函数"""
        msg = "错误消息"
        expected = f"{Colors.BOLD}{Colors.RED}{msg}{Colors.RESET}"
        result = to_code_msg_format(msg)
        self.assertEqual(result, expected)


class TestGetFormattedLogContent(unittest.TestCase):
    """测试获取格式化日志内容的函数"""

    def setUp(self):
        """设置测试环境"""
        # 模拟error_manager.get方法
        self.mock_error_code = MagicMock()
        self.mock_error_code.message = "测试错误消息"
        self.mock_error_code.err_type = ErrorType.UNKNOWN
        self.mock_error_code.faq_url = "https://example.com/faq"

    @patch('ais_bench.benchmark.utils.logger.error_manager')
    def test_get_formatted_log_content_with_unknown_error_type(self, mock_manager):
        """测试处理未知错误类型的情况"""
        mock_manager.get.return_value = self.mock_error_code

        code_str = "TEST-CODE"
        msg = "额外的错误信息"
        result = get_formatted_log_content(code_str, msg)

        # 验证结果中不包含FAQ URL
        self.assertIn(f"[{to_error_code_format(code_str)}]", result)
        self.assertIn(to_code_msg_format("测试错误消息"), result)
        self.assertIn(msg, result)
        self.assertNotIn("Visit", result)

        # 验证调用了error_manager.get
        mock_manager.get.assert_called_once_with(code_str)

    @patch('ais_bench.benchmark.utils.logger.error_manager')
    def test_get_formatted_log_content_with_known_error_type(self, mock_manager):
        """测试处理已知错误类型的情况"""
        # 避免直接修改__ne__方法，而是创建一个自定义的错误类型模拟
        class MockKnownErrorType:
            def __ne__(self, other):
                # 总是返回True，表示这不是UNKNOWN类型
                return True

        self.mock_error_code.err_type = MockKnownErrorType()
        mock_manager.get.return_value = self.mock_error_code

        code_str = "TEST-CODE"
        msg = "额外的错误信息"
        result = get_formatted_log_content(code_str, msg)

        # 验证结果中包含FAQ URL
        self.assertIn(f"[{to_error_code_format(code_str)}]", result)
        self.assertIn(to_code_msg_format("测试错误消息"), result)
        self.assertIn(msg, result)
        self.assertIn(f"Visit {to_url_format(self.mock_error_code.faq_url)}", result)

    @patch('ais_bench.benchmark.utils.logger.error_manager')
    def test_get_formatted_log_content_error_code_not_found(self, mock_manager):
        """测试错误码不存在的情况"""
        mock_manager.get.return_value = None

        code_str = "NONEXISTENT-CODE"
        msg = "任何消息"

        with self.assertRaises(ValueError) as context:
            get_formatted_log_content(code_str, msg)

        self.assertIn(f"error code {code_str} not found", str(context.exception))


class TestColoredLevelFormatter(unittest.TestCase):
    """测试ColoredLevelFormatter类"""

    def setUp(self):
        """设置测试环境"""
        self.formatter = ColoredLevelFormatter()

        # 创建一个更完整的LogRecord模拟对象
        self.mock_record = MagicMock(spec=logging.LogRecord)
        self.mock_record.levelno = logging.INFO
        self.mock_record.levelname = "INFO"
        self.mock_record.getMessage.return_value = "测试消息"
        # 添加必要的属性以避免AttributeError
        self.mock_record.created = time.time()
        self.mock_record.name = "test_logger"
        self.mock_record.pathname = "test.py"
        self.mock_record.lineno = 10

    @patch('logging.Formatter.format')
    def test_format_debug_level(self, mock_super_format):
        """测试格式化DEBUG级别的日志"""
        mock_super_format.return_value = "格式化后的DEBUG消息"
        self.mock_record.levelno = logging.DEBUG

        result = self.formatter.format(self.mock_record)

        # 验证使用了DEBUG格式器
        self.assertEqual(self.formatter._style._fmt, LOG_DEBUG_FORMATTER)
        # 验证调用了父类的format方法
        mock_super_format.assert_called_once_with(self.mock_record)
        # 验证结果
        self.assertEqual(result, "格式化后的DEBUG消息")

    @patch('logging.Formatter.format')
    def test_format_non_debug_level(self, mock_super_format):
        """测试格式化非DEBUG级别的日志"""
        mock_super_format.return_value = "格式化后的非DEBUG消息"

        # 测试INFO级别
        self.mock_record.levelno = logging.INFO
        result = self.formatter.format(self.mock_record)
        self.assertEqual(self.formatter._style._fmt, LOG_NORMAL_FORMATTER)
        self.assertEqual(result, "格式化后的非DEBUG消息")

        # 测试WARNING级别
        mock_super_format.reset_mock()
        self.mock_record.levelno = logging.WARNING
        result = self.formatter.format(self.mock_record)
        self.assertEqual(self.formatter._style._fmt, LOG_NORMAL_FORMATTER)
        self.assertEqual(result, "格式化后的非DEBUG消息")

        # 测试ERROR级别
        mock_super_format.reset_mock()
        self.mock_record.levelno = logging.ERROR
        result = self.formatter.format(self.mock_record)
        self.assertEqual(self.formatter._style._fmt, LOG_NORMAL_FORMATTER)
        self.assertEqual(result, "格式化后的非DEBUG消息")

        # 测试CRITICAL级别
        mock_super_format.reset_mock()
        self.mock_record.levelno = logging.CRITICAL
        result = self.formatter.format(self.mock_record)
        self.assertEqual(self.formatter._style._fmt, LOG_NORMAL_FORMATTER)
        self.assertEqual(result, "格式化后的非DEBUG消息")


class TestAISLogger(unittest.TestCase):
    """测试AISLogger类"""

    def setUp(self):
        """设置测试环境"""
        # 保存原始的stdout和stderr
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        # 替换为StringIO对象以捕获输出
        sys.stdout = StringIO()
        sys.stderr = StringIO()

        # 清除已有的logger处理器，避免影响测试
        for name in logging.root.manager.loggerDict:
            logger = logging.getLogger(name)
            logger.handlers = []

    def tearDown(self):
        """清理测试环境"""
        # 恢复原始的stdout和stderr
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr

    @patch('logging.FileHandler')
    def test_init_main_process_with_log_file(self, mock_file_handler):
        """测试初始化主进程logger并指定日志文件"""
        mock_open_file = mock_open()
        with patch('builtins.open', mock_open_file):
            logger = AISLogger(
                name="test_logger",
                level=logging.INFO,
                is_main_process=True,
                log_file="test.log",
                file_mode='w'
            )

            # 验证创建了FileHandler
            mock_file_handler.assert_called_once_with("test.log", 'w')
            # 验证设置了日志级别
            self.assertEqual(logger.logger.level, logging.INFO)

    @patch('logging.FileHandler')
    def test_init_main_process_without_log_file(self, mock_file_handler):
        """测试初始化主进程logger但不指定日志文件"""
        logger = AISLogger(
            name="test_logger",
            level=logging.INFO,
            is_main_process=True,
            log_file=None
        )

        # 验证没有创建FileHandler
        mock_file_handler.assert_not_called()
        # 验证设置了日志级别
        self.assertEqual(logger.logger.level, logging.INFO)

    @patch('logging.FileHandler')
    def test_init_subprocess(self, mock_file_handler):
        """测试初始化子进程logger"""
        logger = AISLogger(
            name="test_logger",
            level=logging.INFO,
            is_main_process=False
        )

        # 验证没有创建FileHandler
        mock_file_handler.assert_not_called()
        # 验证设置了子进程的日志级别
        from ais_bench.benchmark.utils.logger import SUBPROCESS_LOG_LEVEL
        self.assertEqual(logger.logger.level, SUBPROCESS_LOG_LEVEL)

    @patch('logging.Logger.info')
    def test_info_logging(self, mock_info):
        """测试info日志方法"""
        logger = AISLogger(name="test_logger")
        logger.info("测试信息")
        mock_info.assert_called_once_with("测试信息")

    @patch('logging.Logger.debug')
    def test_debug_logging(self, mock_debug):
        """测试debug日志方法"""
        logger = AISLogger(name="test_logger")
        logger.debug("测试调试信息")
        mock_debug.assert_called_once_with("测试调试信息")

    @patch('logging.Logger.warning')
    def test_warning_logging(self, mock_warning):
        """测试warning日志方法"""
        logger = AISLogger(name="test_logger")
        logger.warning("测试警告信息")
        mock_warning.assert_called_once_with("测试警告信息")

    @patch('ais_bench.benchmark.utils.logger.get_formatted_log_content')
    @patch('logging.Logger.error')
    def test_error_logging(self, mock_error, mock_get_formatted):
        """测试error日志方法"""
        mock_get_formatted.return_value = "格式化后的错误消息"

        logger = AISLogger(name="test_logger")
        logger.error("ERROR-CODE", "错误描述")

        # 验证调用了get_formatted_log_content
        mock_get_formatted.assert_called_once_with("ERROR-CODE", "错误描述")
        # 验证调用了logger.error
        mock_error.assert_called_once_with("格式化后的错误消息")

    @patch('ais_bench.benchmark.utils.logger.get_formatted_log_content')
    @patch('logging.Logger.error')
    def test_error_logging_with_args_kwargs(self, mock_error, mock_get_formatted):
        """测试error日志方法带参数和关键字参数"""
        mock_get_formatted.return_value = "格式化后的错误消息"

        logger = AISLogger(name="test_logger")
        logger.error("ERROR-CODE", "错误描述: %s", "详细信息", exc_info=True)

        # 验证调用了get_formatted_log_content
        mock_get_formatted.assert_called_once_with("ERROR-CODE", "错误描述: %s")
        # 验证调用了logger.error并传递了所有参数
        mock_error.assert_called_once_with("格式化后的错误消息", "详细信息", exc_info=True)


if __name__ == '__main__':
    unittest.main()