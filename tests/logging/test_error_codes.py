import unittest
from typing import Dict
from ais_bench.benchmark.utils.error_codes import (
    ErrorModule, ErrorType, BaseErrorCode, ErrorCodeManager, error_manager
)


class TestErrorModule(unittest.TestCase):
    """测试ErrorModule枚举类"""

    def test_enum_values(self):
        """测试枚举值是否正确"""
        self.assertEqual(ErrorModule.TASK_MANAGER.value, "TMAN")
        self.assertEqual(ErrorModule.PARTITIONER.value, "PARTI")
        self.assertEqual(ErrorModule.SUMMARY.value, "SUMM")
        self.assertEqual(ErrorModule.RUNNER.value, "RUNNER")
        self.assertEqual(ErrorModule.TASK_INFER.value, "TINFER")
        self.assertEqual(ErrorModule.TASK_EVALUATE.value, "TEVAL")
        self.assertEqual(ErrorModule.TASK_MONITOR.value, "TMON")
        self.assertEqual(ErrorModule.TASK_STATUS_MANAGER.value, "TSMAN")
        self.assertEqual(ErrorModule.ICL_INFERENCER.value, "ICLI")
        self.assertEqual(ErrorModule.ICL_EVALUATOR.value, "ICLE")
        self.assertEqual(ErrorModule.ICL_RETRIEVER.value, "ICLR")
        self.assertEqual(ErrorModule.MODEL.value, "MODEL")
        self.assertEqual(ErrorModule.UTILS.value, "UTILS")
        self.assertEqual(ErrorModule.UNKNOWN.value, "UNK")

    def test_enum_members_count(self):
        """测试枚举成员数量是否正确"""
        self.assertEqual(len(ErrorModule), 14)


class TestErrorType(unittest.TestCase):
    """测试ErrorType枚举类"""

    def test_enum_values(self):
        """测试枚举值是否正确"""
        self.assertEqual(ErrorType.UNKNOWN.value, "UNK")

    def test_enum_members_count(self):
        """测试枚举成员数量是否正确"""
        self.assertEqual(len(ErrorType), 1)


class TestBaseErrorCode(unittest.TestCase):
    """测试BaseErrorCode类"""

    def setUp(self):
        """设置测试环境"""
        self.module = ErrorModule.UTILS
        self.err_type = ErrorType.UNKNOWN
        self.code = 42
        self.message = "测试错误消息"
        self.error_code = BaseErrorCode(
            module=self.module, err_type=self.err_type, code=self.code,
            message=self.message
        )

    def test_init(self):
        """测试初始化方法是否正确设置属性"""
        self.assertEqual(self.error_code.module, self.module)
        self.assertEqual(self.error_code.err_type, self.err_type)
        self.assertEqual(self.error_code.code, self.code)
        self.assertEqual(self.error_code.message, self.message)

    def test_full_code(self):
        """测试full_code属性是否正确生成完整错误码"""
        expected_full_code = f"{self.module.value}-{self.err_type.value}-{self.code:03d}"
        self.assertEqual(self.error_code.full_code, expected_full_code)

        # 测试不同代码长度的格式化
        error_code_1 = BaseErrorCode(self.module, self.err_type, 1, self.message)
        self.assertEqual(error_code_1.full_code, f"{self.module.value}-{self.err_type.value}-001")

        error_code_10 = BaseErrorCode(self.module, self.err_type, 10, self.message)
        self.assertEqual(error_code_10.full_code, f"{self.module.value}-{self.err_type.value}-010")

    def test_str_representation(self):
        """测试字符串表示是否正确"""
        expected_str = f"{self.error_code.full_code}: {self.message}"
        self.assertEqual(str(self.error_code), expected_str)

    def test_faq_url(self):
        """测试FAQ URL是否正确生成"""
        expected_url = f"{BaseErrorCode.FAQ_BASE_URL}{self.error_code.full_code}"
        self.assertEqual(self.error_code.faq_url, expected_url)


class TestErrorCodeManager(unittest.TestCase):
    """测试ErrorCodeManager类"""

    def setUp(self):
        """设置测试环境"""
        self.manager = ErrorCodeManager()
        self.error_code_1 = BaseErrorCode(
            ErrorModule.UTILS, ErrorType.UNKNOWN, 1, "测试错误1"
        )
        self.error_code_2 = BaseErrorCode(
            ErrorModule.RUNNER, ErrorType.UNKNOWN, 2, "测试错误2"
        )

    def test_init(self):
        """测试初始化方法是否正确创建空的错误码字典"""
        self.assertIsInstance(self.manager._error_codes, Dict)
        self.assertEqual(len(self.manager._error_codes), 0)

    def test_register(self):
        """测试注册错误码方法"""
        # 测试成功注册
        self.manager.register(self.error_code_1)
        self.assertIn(self.error_code_1.full_code, self.manager._error_codes)
        self.assertEqual(self.manager._error_codes[self.error_code_1.full_code], self.error_code_1)

        # 测试注册多个错误码
        self.manager.register(self.error_code_2)
        self.assertIn(self.error_code_1.full_code, self.manager._error_codes)
        self.assertIn(self.error_code_2.full_code, self.manager._error_codes)
        self.assertEqual(len(self.manager._error_codes), 2)

    def test_register_duplicate(self):
        """测试注册重复错误码时是否抛出异常"""
        self.manager.register(self.error_code_1)
        with self.assertRaises(ValueError) as context:
            self.manager.register(self.error_code_1)
        self.assertIn(f"error code {self.error_code_1.full_code} is exist!", str(context.exception))

    def test_get(self):
        """测试获取错误码方法"""
        # 测试获取已注册的错误码
        self.manager.register(self.error_code_1)
        retrieved_error = self.manager.get(self.error_code_1.full_code)
        self.assertEqual(retrieved_error, self.error_code_1)

        # 测试获取未注册的错误码
        non_existent_code = "NONEXISTENT-CODE"
        retrieved_error = self.manager.get(non_existent_code)
        self.assertIsNone(retrieved_error)

    def test_list_all(self):
        """测试列出所有错误码方法"""
        # 测试空管理器
        all_errors = self.manager.list_all()
        self.assertEqual(len(all_errors), 0)

        # 测试注册错误码后列出所有错误码
        self.manager.register(self.error_code_1)
        self.manager.register(self.error_code_2)
        all_errors = self.manager.list_all()
        self.assertEqual(len(all_errors), 2)
        self.assertIn(self.error_code_1.full_code, all_errors)
        self.assertIn(self.error_code_2.full_code, all_errors)

        # 测试返回的是副本，修改返回值不会影响原始字典
        all_errors_copy = self.manager.list_all()
        all_errors_copy.pop(self.error_code_1.full_code)
        self.assertEqual(len(self.manager.list_all()), 2)  # 原始字典未受影响


class TestGlobalErrorManager(unittest.TestCase):
    """测试全局error_manager对象"""

    def test_error_manager_instance(self):
        """测试error_manager是否为ErrorCodeManager的实例"""
        self.assertIsInstance(error_manager, ErrorCodeManager)

    def test_pre_registered_errors(self):
        """测试预注册的错误码是否存在"""
        # 测试至少有一个错误码被预注册
        all_errors = error_manager.list_all()
        self.assertGreater(len(all_errors), 0)

        # 测试一些特定的预注册错误码
        # 这里仅验证存在性，不检查具体数量，因为预注册错误码可能会随时间变化
        error_codes_to_check = [
            "TMAN-UNK-001",  # TaskManager相关
            "PARTI-UNK-001", # Partitioner相关
            "SUMM-UNK-001",  # Summary相关
            "RUNNER-UNK-001", # Runner相关
            "UNK-UNK-001"    # 未知错误
        ]

        found = False
        for code in error_codes_to_check:
            if error_manager.get(code):
                found = True
                break
        self.assertTrue(found, "没有找到任何预注册的错误码")


if __name__ == '__main__':
    unittest.main()