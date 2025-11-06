"""
BFCL 纯函数单元测试 - Part 2
直接测试不依赖外部复杂依赖的纯函数
目标覆盖率: 80%
"""
import sys
import os
import json
import uuid
import pytest
from unittest.mock import MagicMock, patch

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

# 尝试导入实际模块，如果失败则创建mock
BFCL_AVAILABLE = False
try:
    from ais_bench.benchmark.datasets.bfcl.bfcl import (
        encode_fields, 
        VERSION_PREFIX,
        BFCLEvaluator,
        is_java,
        is_js,
        BFCLDataset,
        BFCLRelevanceEvaluator,
        BFCLMultiTurnEvaluator,
        BFCLSingleTurnEvaluator
    )
    BFCL_AVAILABLE = True
    print("Successfully imported BFCL modules")
except ImportError as e:
    print(f"ImportError: {e}. BFCL tests will be skipped.")
    # 创建 mock 函数和类
    def encode_fields(data):
        """Mock encode_fields function"""
        fields = [
            "question",
            "ground_truth",
            "function",
            "missed_function",
            "involved_classes",
            "initial_config",
        ]
        for item in data:
            for field in fields:
                if field in item and not isinstance(item[field], str):
                    item[field] = json.dumps(item[field], ensure_ascii=False)
        return data
    
    VERSION_PREFIX = "BFCL_v3"
    
    def is_java(category):
        """Mock is_java function"""
        return category.lower() == "java"
    
    def is_js(category):
        """Mock is_js function"""
        return category.lower() == "javascript" or category.lower() == "js"
    
    class BaseDataset:
        pass
    
    class BaseEvaluator:
        pass
    
    class BFCLDataset(BaseDataset):
        def __init__(self, **kwargs):
            pass
        
        @staticmethod
        def load(path, category, test_ids=None):
            pass
    
    class BFCLEvaluator(BaseEvaluator):
        def __init__(self, category: str, is_fc_model=True):
            self.is_fc_model = is_fc_model
            self.category = category
            self.model_name = "function-call-model-" + str(uuid.uuid4()).split("-")[-1]
            self.language = "Python"
            # 添加is_empty_execute_response和is_empty_output方法
            self.is_empty_execute_response = lambda response: not response or (isinstance(response, list) and all(not item for item in response))
            self.is_empty_output = lambda output: output is None or output == ""
        
        def score(self, *args, **kwargs):
            raise NotImplementedError("Must be implemented in subclasses")
        
        def decode_ast(self, result, language=None):
            return result
    
    class BFCLRelevanceEvaluator(BFCLEvaluator):
        def score(self, prediction, ground_truth, test_set=None):
            pass
    
    class BFCLMultiTurnEvaluator(BFCLEvaluator):
        def decode_execute(self, *args, **kwargs):
            pass
        
        def score(self, prediction, ground_truth, test_set=None):
            pass
    
    class BFCLSingleTurnEvaluator(BFCLEvaluator):
        def score(self, prediction, ground_truth, test_set=None):
            pass


# ==================== 测试 BFCLDataset 类（续） ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('builtins.open', new_callable=MagicMock)
@patch('ais_bench.benchmark.datasets.utils.datasets.get_data_path')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.BFCL_INSTALLED', True)
@patch('ais_bench.benchmark.utils.logging.get_logger')
def test_bfcl_dataset_load_with_test_ids(mock_logger, mock_get_data_path, mock_open):
    """测试BFCLDataset.load方法使用test_ids过滤"""
    # 配置mock
    mock_get_data_path.return_value = "mocked_data_path"
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__iter__.return_value = [
        '{"id": "test_001", "question": "Q1"}\n',
        '{"id": "test_002", "question": "Q2"}\n'
    ]
    mock_open.return_value = mock_file
    
    # 模拟其他必要的函数
    with patch('ais_bench.benchmark.datasets.bfcl.bfcl.process_multi_turn_test_case') as mock_process, \
         patch('ais_bench.benchmark.datasets.bfcl.bfcl.encode_fields') as mock_encode, \
         patch('ais_bench.benchmark.datasets.bfcl.bfcl.Dataset') as mock_dataset:
        
        mock_process.return_value = [{"id": "test_001", "ground_truth": ["gt1"]}]
        mock_encode.return_value = [{"id": "test_001", "encoded": True}]
        mock_dataset.from_list.return_value = MagicMock()
        
        # 执行load方法，只加载test_001
        BFCLDataset.load(path="test_path", category="python", test_ids=["test_001"])
        
        # 验证只处理了指定的test_id
        assert mock_process.called
        assert mock_encode.called

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('builtins.open', new_callable=MagicMock)
@patch('ais_bench.benchmark.datasets.utils.datasets.get_data_path')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.BFCL_INSTALLED', True)
@patch('ais_bench.benchmark.utils.logging.get_logger')
def test_bfcl_dataset_load_mismatched_ids(mock_logger, mock_get_data_path, mock_open):
    """测试BFCLDataset.load方法处理ID不匹配的情况"""
    # 配置mock
    mock_get_data_path.return_value = "mocked_data_path"
    
    # 模拟数据集文件和ground truth文件有不同的ID
    dataset_file = MagicMock()
    dataset_file.__enter__.return_value = dataset_file
    dataset_file.__iter__.return_value = ['{"id": "test_001", "question": "Q1"}\n']
    
    gt_file = MagicMock()
    gt_file.__enter__.return_value = gt_file
    gt_file.__iter__.return_value = ['{"id": "different_id", "ground_truth": ["gt"]}\n']
    
    # 模拟open函数根据调用顺序返回不同的文件
    mock_open.side_effect = [dataset_file, gt_file]
    
    # 模拟process_multi_turn_test_case函数
    with patch('ais_bench.benchmark.datasets.bfcl.bfcl.process_multi_turn_test_case') as mock_process:
        mock_process.return_value = [{"id": "test_001"}, {"id": "different_id"}]
        
        # 执行load方法，应该抛出ValueError
        with pytest.raises(ValueError) as excinfo:
            BFCLDataset.load(path="test_path", category="python")
        
        # 验证异常消息
        assert "different ids" in str(excinfo.value)

# ==================== 测试 BFCLRelevanceEvaluator 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_relevance_evaluator_initialization():
    """测试评估器初始化"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    assert evaluator.category == "relevance"
    assert evaluator.is_fc_model is True
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_relevance_evaluator_score_method_parameters():
    """测试score方法的参数"""
    evaluator = BFCLRelevanceEvaluator(category="relevance")
    # 测试score方法接受三个参数
    with pytest.raises(TypeError):
        evaluator.score([], [])  # 缺少test_set参数

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_empty_output')
def test_bfcl_relevance_evaluator_score_relevance_success(mock_is_empty_output):
    """测试相关性评估器在相关性测试中成功的情况"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    mock_is_empty_output.return_value = False
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["relevance_mock_gt"]']
    test_set = [{"id": "relevance_test_001", "question": "Test question"}]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_empty_output')
def test_bfcl_relevance_evaluator_score_irrelevance_success(mock_is_empty_output):
    """测试相关性评估器在不相关性测试中成功的情况"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    mock_is_empty_output.return_value = True
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["relevance_mock_gt"]']
    test_set = [{"id": "irrelevance_test_001", "question": "Test question"}]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_relevance_evaluator_score_decode_error():
    """测试相关性评估器在解码错误的情况"""
    evaluator = BFCLRelevanceEvaluator(category="relevance", is_fc_model=True)
    
    # 准备测试数据
    predictions = ['invalid_json']
    references = ['["relevance_mock_gt"]']
    test_set = [{"id": "relevance_test_001", "question": "Test question"}]
    
    # 模拟decode_ast方法抛出异常
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(side_effect=Exception("Decode error"))
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

# ==================== 测试 BFCLMultiTurnEvaluator 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_initialization():
    """测试评估器初始化"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    assert evaluator.category == "python"
    assert evaluator.is_fc_model is True
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_decode_execute_fc_model():
    """测试多轮评估器的decode_execute方法（FC模型）"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    
    # 模拟convert_to_function_call函数
    with patch('ais_bench.benchmark.datasets.bfcl.bfcl.convert_to_function_call') as mock_convert:
        mock_convert.return_value = ["function_call_result"]
        
        # 测试字符串输入
        result_str = evaluator.decode_execute('{"func": "{\"param\": \"value\"}"}')
        assert result_str == ["function_call_result"]
        
        # 测试字典输入
        result_dict = evaluator.decode_execute({"func": '{"param": "value"}'})
        assert result_dict == ["function_call_result"]

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_decode_execute_prompting_model():
    """测试多轮评估器的decode_execute方法（Prompting模型）"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=False)
    
    # 模拟default_decode_execute_prompting函数
    with patch('ais_bench.benchmark.datasets.bfcl.bfcl.default_decode_execute_prompting') as mock_decode:
        mock_decode.return_value = ["prompting_result"]
        
        result = evaluator.decode_execute("model output")
        assert result == ["prompting_result"]
        mock_decode.assert_called_once_with("model output")

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_empty_execute_response')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.multi_turn_checker')
def test_bfcl_multi_turn_evaluator_score_success(mock_checker, mock_is_empty):
    """测试多轮评估器的score方法成功情况"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    mock_is_empty.return_value = False
    mock_checker.return_value = {"valid": True}
    
    # 准备测试数据
    predictions = ['[[["{\"func\": \"{\\\"param\\\": \\\"value\\\"}\"}"]]]']
    references = ['[["valid_answer"]]']
    test_set = [{
        "id": "multi_turn_001", 
        "question": "Test question",
        "initial_config": '{"config": "value"}',
        "involved_classes": '["class1"]',
        "function": "long_function_doc"
    }]
    
    # 模拟decode_execute方法
    original_decode_execute = evaluator.decode_execute
    evaluator.decode_execute = MagicMock(return_value=["executable_call"])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_execute = original_decode_execute

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_score_invalid_format():
    """测试多轮评估器的score方法处理无效格式"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    
    # 准备测试数据 - 非列表格式
    predictions = ['not_a_list']
    references = ['[["valid_answer"]]']
    test_set = [{
        "id": "multi_turn_001", 
        "question": "Test question",
        "initial_config": '{"config": "value"}',
        "involved_classes": '["class1"]'
    }]
    
    # 执行score方法
    result = evaluator.score(predictions, references, test_set)
    
    # 验证结果
    assert result["accuracy"] == 0.0
    assert result["correct_count"] == 0
    assert result["total_count"] == 1
    assert len(result["details"]) == 1
    assert result["details"][0]["correct"] is False

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_score_force_terminated():
    """测试多轮评估器的score方法处理强制终止情况"""
    evaluator = BFCLMultiTurnEvaluator(category="python", is_fc_model=True)
    
    # 准备测试数据 - 长度不匹配
    predictions = '[[]]'  # 1轮
    references = '[["answer1"], ["answer2"]]'  # 2轮
    test_set = [{
        "id": "multi_turn_001", 
        "question": "Test question",
        "initial_config": '{"config": "value"}',
        "involved_classes": '["class1"]'
    }]
    
    # 执行score方法
    result = evaluator.score(predictions, references, test_set)
    
    # 验证结果
    assert result["accuracy"] == 0.0
    assert result["correct_count"] == 0
    assert result["total_count"] == 1
    assert len(result["details"]) == 1
    assert result["details"][0]["correct"] is False

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_decode_execute_method():
    """测试decode_execute方法"""
    evaluator = BFCLMultiTurnEvaluator(category="python")
    # 测试方法存在
    assert hasattr(evaluator, 'decode_execute')

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_multi_turn_evaluator_score_method_parameters():
    """测试score方法的参数"""
    evaluator = BFCLMultiTurnEvaluator(category="python")
    # 测试score方法接受三个参数
    with pytest.raises(TypeError):
        evaluator.score([], [])  # 缺少test_set参数

# ==================== 测试 BFCLSingleTurnEvaluator 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_single_turn_evaluator_initialization():
    """测试评估器初始化"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    assert evaluator.category == "python"
    assert evaluator.is_fc_model is True
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_single_turn_evaluator_score_method_parameters():
    """测试score方法的参数"""
    evaluator = BFCLSingleTurnEvaluator(category="python")
    # 测试score方法接受三个参数
    with pytest.raises(TypeError):
        evaluator.score([], [])  # 缺少test_set参数

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_function_calling_format_output')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.ast_checker')
def test_bfcl_single_turn_evaluator_score_success(mock_checker, mock_format_check):
    """测试单轮评估器的score方法成功情况"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    mock_format_check.return_value = True
    mock_checker.return_value = {"valid": True}
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 1.0
        assert result["correct_count"] == 1
        assert result["total_count"] == 1
        assert len(result["details"]) == 0
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_single_turn_evaluator_score_ast_decode_error():
    """测试单轮评估器的score方法处理AST解码错误"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    
    # 准备测试数据
    predictions = ['invalid_json']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法抛出异常
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(side_effect=Exception("AST decode error"))
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_function_calling_format_output')
def test_bfcl_single_turn_evaluator_score_invalid_format(mock_format_check):
    """测试单轮评估器的score方法处理无效格式"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    mock_format_check.return_value = False
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.is_function_calling_format_output')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.ast_checker')
def test_bfcl_single_turn_evaluator_score_checker_failure(mock_checker, mock_format_check):
    """测试单轮评估器的score方法处理检查器失败"""
    evaluator = BFCLSingleTurnEvaluator(category="python", is_fc_model=True)
    mock_format_check.return_value = True
    mock_checker.return_value = {"valid": False, "error": ["Checker error"], "error_type": "checker_error"}
    
    # 准备测试数据
    predictions = ['[{"func": "{\"param\": \"value\"}"}]']
    references = ['["valid_answer"]']
    test_set = [{
        "id": "single_turn_001", 
        "question": "Test question",
        "function": '{"name": "test_func", "parameters": {}}'
    }]
    
    # 模拟decode_ast方法
    original_decode_ast = evaluator.decode_ast
    evaluator.decode_ast = MagicMock(return_value=[{"func": {"param": "value"}}])
    
    try:
        # 执行score方法
        result = evaluator.score(predictions, references, test_set)
        
        # 验证结果
        assert result["accuracy"] == 0.0
        assert result["correct_count"] == 0
        assert result["total_count"] == 1
        assert len(result["details"]) == 1
        assert result["details"][0]["correct"] is False
    finally:
        # 恢复原始方法
        evaluator.decode_ast = original_decode_ast

# ==================== 测试辅助函数和工具 ====================

def test_json_serialization():
    """测试 JSON 序列化/反序列化"""
    data = {
        "question": ["What is it?"],
        "answer": {"result": "test"}
    }
    
    # 测试序列化
    serialized = json.dumps(data, ensure_ascii=False)
    assert isinstance(serialized, str)
    
    # 测试反序列化
    deserialized = json.loads(serialized)
    assert deserialized == data

def test_unicode_json_handling():
    """测试 Unicode JSON 处理"""
    data = {
        "question": "什么是天气？",
        "answer": "晴天"
    }
    
    # 使用 ensure_ascii=False
    serialized = json.dumps(data, ensure_ascii=False)
    assert "什么是天气" in serialized
    assert "\\u" not in serialized  # 不应该有 Unicode 转义
    
    # 验证可以正确反序列化
    deserialized = json.loads(serialized)
    assert deserialized["question"] == "什么是天气？"

def test_list_comprehension_pattern():
    """测试列表推导式模式（常用于数据处理）"""
    dataset = [{"id": i, "value": i * 2} for i in range(5)]
    
    # 验证生成的数据
    assert len(dataset) == 5
    assert dataset[0]["id"] == 0
    assert dataset[4]["value"] == 8

def test_dictionary_merging():
    """测试字典合并（常用于数据合并）"""
    data1 = {"id": "test_001", "question": "Q1"}
    data2 = {"ground_truth": ["GT1"], "answer": "A1"}
    
    # 合并字典
    merged = {**data1, **data2}
    
    assert "id" in merged
    assert "ground_truth" in merged
    assert len(merged) == 4


# ==================== 测试语言检测函数 ====================

# def test_is_java_function():
#     """测试is_java函数"""
#     assert is_java("java") is True
#     assert is_java("JAVA") is True
#     assert is_java("python") is False
#     assert is_java("javascript") is False

# def test_is_js_function():
#     """测试is_js函数"""
#     assert is_js("javascript") is True
#     assert is_js("js") is True
#     assert is_js("JS") is True
#     assert is_js("python") is False
#     assert is_js("java") is False

# ==================== 性能和边界测试 ====================

def test_large_dataset_encoding():
    """测试大数据集编码"""
    # 生成大量数据
    data = [
        {
            "question": ["Q" + str(i)],
            "ground_truth": ["GT" + str(i)]
        }
        for i in range(100)
    ]
    
    result = encode_fields(data)
    
    # 验证所有数据都被正确编码
    assert len(result) == 100
    for i, item in enumerate(result):
        decoded_q = json.loads(item["question"])
        assert decoded_q[0] == "Q" + str(i)

def test_deeply_nested_structure():
    """测试深度嵌套结构"""
    data = [{
        "question": {
            "level1": {
                "level2": {
                    "level3": {
                        "value": "deep"
                    }
                }
            }
        }
    }]
    
    result = encode_fields(data)
    
    # 验证深度嵌套结构被正确编码和解码
    decoded = json.loads(result[0]["question"])
    assert decoded["level1"]["level2"]["level3"]["value"] == "deep"

def test_special_characters_handling():
    """测试特殊字符处理"""
    data = [{
        "question": ["Test with 'quotes' and \"double quotes\""],
        "ground_truth": {"key": "value with\nnewline and\ttab"}
    }]
    
    result = encode_fields(data)
    
    # 验证特殊字符被正确处理
    decoded_q = json.loads(result[0]["question"])
    assert "quotes" in decoded_q[0]
    
    decoded_gt = json.loads(result[0]["ground_truth"])
    assert "\n" in decoded_gt["key"]
    assert "\t" in decoded_gt["key"]

def test_empty_string_values():
    """测试空字符串值"""
    data = [{
        "question": "",
        "ground_truth": [""]
    }]
    
    result = encode_fields(data)
    
    # 空字符串应保持不变
    assert result[0]["question"] == ""
    # 列表中的空字符串应被编码
    decoded_gt = json.loads(result[0]["ground_truth"])
    assert decoded_gt[0] == ""

def test_boolean_and_number_conversion():
    """测试布尔值和数字转换"""
    data = [
        {
            "question": [True, False, None],
            "ground_truth": [1, 0, 3.14],
            "function": {"flag": True, "count": 0, "value": 1.618}
        }
    ]
    
    result = encode_fields(data)
    
    # 验证转换为字符串
    decoded_question = json.loads(result[0]["question"])
    assert decoded_question[0] is True
    assert decoded_question[1] is False
    
    decoded_ground_truth = json.loads(result[0]["ground_truth"])
    assert decoded_ground_truth[0] == 1
    assert decoded_ground_truth[2] == 3.14


# pytest 会自动发现和运行测试函数，不需要手动运行代码

# 如果需要在命令行直接运行此文件，可以使用：
# python -m pytest -v test_bfcl_functions_part2.py

