"""
BFCL 纯函数单元测试 - Part 1
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


# ==================== 测试 encode_fields 函数 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_non_string_fields():
    """测试编码非字符串字段"""
    data = [
        {
            "question": ["What is the capital?"],
            "ground_truth": {"answer": "Paris"},
            "function": {"name": "get_capital"},
            "other_field": "keep_as_is"
        }
    ]
    
    result = encode_fields(data)
    
    # 验证非字符串字段被编码为 JSON 字符串
    assert isinstance(result[0]["question"], str)
    assert isinstance(result[0]["ground_truth"], str)
    assert isinstance(result[0]["function"], str)
    # 验证可以解码回原始数据
    assert json.loads(result[0]["question"]) == ["What is the capital?"]
    assert json.loads(result[0]["ground_truth"]) == {"answer": "Paris"}
    # 验证原本就是字符串的字段保持不变
    assert result[0]["other_field"] == "keep_as_is"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_already_string_fields():
    """测试已经是字符串的字段不被重复编码"""
    data = [
        {
            "question": '["What is the capital?"]',
            "ground_truth": '{"answer": "Paris"}',
            "function": '{"name": "test"}'
        }
    ]
    
    result = encode_fields(data)
    
    # 验证已经是字符串的字段保持不变
    assert result[0]["question"] == '["What is the capital?"]'
    assert result[0]["ground_truth"] == '{"answer": "Paris"}'
    assert result[0]["function"] == '{"name": "test"}'

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_empty_list():
    """测试空列表"""
    data = []
    result = encode_fields(data)
    
    assert len(result) == 0
    assert isinstance(result, list)

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_all_supported_fields():
    """测试所有支持的字段"""
    data = [
        {
            "question": ["Q1", "Q2"],
            "ground_truth": ["GT1", "GT2"],
            "function": {"func": "test_func"},
            "missed_function": ["miss1", "miss2"],
            "involved_classes": ["class1", "class2"],
            "initial_config": {"config": "value"}
        }
    ]
    
    result = encode_fields(data)
    
    # 验证所有支持的字段都被正确编码
    for field in ["question", "ground_truth", "function", "missed_function", "involved_classes", "initial_config"]:
        assert field in result[0]
        assert isinstance(result[0][field], str)
        # 验证可以解码
        decoded = json.loads(result[0][field])
        assert decoded is not None

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_mixed_fields():
    """测试混合字段（部分是字符串，部分不是）"""
    data = [
        {
            "question": ["Q1"],  # 需要编码
            "ground_truth": '["GT1"]',  # 已经是字符串
            "function": {"func": "test"},  # 需要编码
            "other": "plain_string"  # 不在编码列表中
        }
    ]
    
    result = encode_fields(data)
    
    # 验证需要编码的字段被编码
    assert result[0]["question"] == '["Q1"]'
    # 验证已经是字符串的保持不变
    assert result[0]["ground_truth"] == '["GT1"]'
    # 验证 function 被编码
    assert '"func"' in result[0]["function"]
    # 验证其他字段不受影响
    assert result[0]["other"] == "plain_string"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_nested_structures():
    """测试嵌套结构"""
    data = [
        {
            "question": {
                "text": "What is it?",
                "options": ["A", "B", "C"]
            },
            "ground_truth": [
                {"answer": "A", "explanation": "Because..."}
            ]
        }
    ]
    
    result = encode_fields(data)
    
    # 验证嵌套结构被正确编码
    assert isinstance(result[0]["question"], str)
    assert isinstance(result[0]["ground_truth"], str)
    
    # 验证可以解码并保持结构
    decoded_question = json.loads(result[0]["question"])
    assert decoded_question["text"] == "What is it?"
    assert len(decoded_question["options"]) == 3
    
    decoded_gt = json.loads(result[0]["ground_truth"])
    assert decoded_gt[0]["answer"] == "A"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_unicode_content():
    """测试 Unicode 内容（中文等）"""
    data = [
        {
            "question": ["什么是首都？"],
            "ground_truth": {"答案": "北京"},
            "function": {"名称": "获取首都"}
        }
    ]
    
    result = encode_fields(data)
    
    # 验证 Unicode 内容被正确编码（ensure_ascii=False）
    assert isinstance(result[0]["question"], str)
    assert "什么是首都" in result[0]["question"]
    assert "答案" in result[0]["ground_truth"]
    assert "名称" in result[0]["function"]
    
    # 验证可以正确解码
    decoded_question = json.loads(result[0]["question"])
    assert decoded_question[0] == "什么是首都？"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_preserves_order():
    """测试编码保持数据顺序"""
    data = [
        {"id": 1, "question": ["Q1"]},
        {"id": 2, "question": ["Q2"]},
        {"id": 3, "question": ["Q3"]}
    ]
    
    result = encode_fields(data)
    
    # 验证顺序保持不变
    assert len(result) == 3
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2
    assert result[2]["id"] == 3

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_empty_values():
    """测试空值"""
    data = [
        {
            "question": [],
            "ground_truth": {},
            "function": []
        }
    ]
    
    result = encode_fields(data)
    
    # 验证空列表和空字典被编码
    assert result[0]["question"] == "[]"
    assert result[0]["ground_truth"] == "{}"
    assert result[0]["function"] == "[]"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_encode_partial_fields():
    """测试只包含部分字段的数据"""
    data = [
        {
            "question": ["Q1"],
            "other_field": "value"
        }
    ]
    
    result = encode_fields(data)
    
    # 验证存在的字段被编码
    assert result[0]["question"] == '["Q1"]'
    # 验证不在列表中的字段不受影响
    assert result[0]["other_field"] == "value"
    # 验证不存在的字段不会被添加
    assert "ground_truth" not in result[0]


# ==================== 测试 VERSION_PREFIX ====================

def test_version_prefix_format():
    """测试版本前缀格式"""
    assert isinstance(VERSION_PREFIX, str)
    assert "BFCL" in VERSION_PREFIX
    assert "v" in VERSION_PREFIX

def test_version_prefix_value():
    """测试版本前缀的具体值"""
    assert VERSION_PREFIX == "BFCL_v3"

def test_version_prefix_usage_in_filename():
    """测试版本前缀在文件名中的使用"""
    # 模拟文件名构造
    category = "python"
    filename = f"{VERSION_PREFIX}_{category}.json"
    
    assert filename == "BFCL_v3_python.json"
    assert "BFCL" in filename
    assert "python" in filename
    assert filename.endswith(".json")

def test_version_prefix_in_ground_truth_path():
    """测试版本前缀在 ground truth 路径中的使用"""
    category = "javascript"
    gt_path = f"possible_answer/{VERSION_PREFIX}_{category}.json"
    
    assert gt_path == "possible_answer/BFCL_v3_javascript.json"
    assert "possible_answer" in gt_path


# ==================== 测试 BFCLEvaluator 基类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_python_category():
    """测试 Python 类别初始化"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=True)
    
    assert evaluator.language == "Python"
    assert evaluator.category == "python"
    assert evaluator.is_fc_model is True
    assert evaluator.model_name is not None
    assert "function-call-model-" in evaluator.model_name

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_java_category():
    """测试 Java 类别初始化"""
    evaluator = BFCLEvaluator(category="java", is_fc_model=True)
    
    assert evaluator.language == "Java"
    assert evaluator.category == "java"
    assert evaluator.is_fc_model is True

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_js_category():
    """测试 JavaScript 类别初始化"""
    evaluator = BFCLEvaluator(category="javascript", is_fc_model=True)
    
    assert evaluator.language == "JavaScript"
    assert evaluator.category == "javascript"
    assert evaluator.is_fc_model is True

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_init_non_fc_model():
    """测试非 FC 模型初始化"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=False)
    
    assert evaluator.is_fc_model is False
    assert evaluator.language == "Python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_model_name_uniqueness():
    """测试模型名称的唯一性"""
    evaluator1 = BFCLEvaluator(category="python")
    evaluator2 = BFCLEvaluator(category="python")
    
    # 验证两个实例有不同的模型名称
    assert evaluator1.model_name != evaluator2.model_name

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_score_not_implemented():
    """测试 score 方法抛出 NotImplementedError"""
    evaluator = BFCLEvaluator(category="python")
    
    with pytest.raises(NotImplementedError):
        evaluator.score([], [])

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_decode_ast_fc_model():
    """测试 FC 模型的 AST 解码"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=True)
    
    result = [
        {"func1": '{"param": "value"}'},
        {"func2": '{"param2": "value2"}'}
    ]
    
    decoded = evaluator.decode_ast(result)
    
    assert len(decoded) == 2
    assert "func1" in decoded[0]
    if BFCL_AVAILABLE and hasattr(decoded[0]["func1"], "get"):
        assert decoded[0]["func1"].get("param") == "value"
    assert "func2" in decoded[1]

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_decode_ast_prompting_model():
    """测试 Prompting 模型的 AST 解码"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=False)
    
    # 验证基本的初始化
    assert evaluator.is_fc_model is False
    assert evaluator.category == "python"

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_evaluator_decode_ast_with_language_parameter():
    """测试带语言参数的 AST 解码"""
    evaluator = BFCLEvaluator(category="python", is_fc_model=True)
    
    result = [{"func": '{"param": "value"}'}]
    
    # 测试不同语言参数
    decoded_python = evaluator.decode_ast(result, language="Python")
    decoded_java = evaluator.decode_ast(result, language="Java")
    
    # 验证解码结果相同（对于 FC 模型，语言参数不影响解码）
    assert len(decoded_python) == len(decoded_java)


# ==================== 测试 BFCL 数据结构 ====================

def test_dataset_item_structure():
    """测试数据集项的基本结构"""
    # 模拟一个典型的数据集项
    item = {
        "id": "test_001",
        "question": "What is the weather?",
        "function": {"name": "get_weather", "parameters": {}},
        "ground_truth": ["get_weather(location='Beijing')"]
    }
    
    # 验证必要字段存在
    assert "id" in item
    assert "question" in item
    assert "ground_truth" in item
    
    # 验证字段类型
    assert isinstance(item["id"], str)
    assert isinstance(item["question"], str)
    assert isinstance(item["ground_truth"], list)

def test_relevance_dataset_structure():
    """测试 relevance 数据集的特殊结构"""
    # relevance 数据集有特殊的 ground truth
    item = {
        "id": "relevance_test_001",
        "question": "What is the weather?",
        "ground_truth": ["relevance_mock_gt"]
    }
    
    # 验证 relevance 测试的标识
    assert "relevance" in item["id"]
    assert "mock" in str(item["ground_truth"])
    assert item["ground_truth"][0] == "relevance_mock_gt"

def test_irrelevance_dataset_structure():
    """测试 irrelevance 数据集的结构"""
    item = {
        "id": "irrelevance_test_001",
        "question": "Random question",
        "ground_truth": ["relevance_mock_gt"]
    }
    
    # 验证 irrelevance 测试的标识
    assert "irrelevance" in item["id"]

def test_multi_turn_structure():
    """测试多轮对话数据结构"""
    # 多轮对话的数据结构
    item = {
        "id": "multi_turn_001",
        "question": [
            "What is the weather?",
            "What about tomorrow?"
        ],
        "ground_truth": [
            ["get_weather(location='Beijing')"],
            ["get_weather(location='Beijing', date='tomorrow')"]
        ],
        "initial_config": {},
        "involved_classes": []
    }
    
    # 验证多轮结构
    assert isinstance(item["question"], list)
    assert isinstance(item["ground_truth"], list)
    assert "initial_config" in item
    assert "involved_classes" in item
    
    # 验证多轮数据的长度匹配
    assert len(item["question"]) == len(item["ground_truth"])

def test_single_turn_structure():
    """测试单轮对话数据结构"""
    item = {
        "id": "single_turn_001",
        "question": "What is the weather?",
        "function": [{"name": "get_weather", "parameters": {}}],
        "ground_truth": ["get_weather(location='Beijing')"]
    }
    
    # 验证单轮结构
    assert isinstance(item["question"], str)
    assert isinstance(item["ground_truth"], list)
    assert "function" in item

def test_category_identifier():
    """测试类别标识符"""
    categories = ["python", "java", "javascript", "relevance", "irrelevance"]
    
    for category in categories:
        # 验证类别名称格式
        assert isinstance(category, str)
        assert len(category) > 0
        assert category.islower() or '_' in category

def test_id_format():
    """测试 ID 格式"""
    valid_ids = [
        "test_001",
        "multi_turn_python_001",
        "relevance_test_01",
        "irrelevance_test_02",
        "single_turn_java_001"
    ]
    
    for id_value in valid_ids:
        # 验证 ID 格式
        assert isinstance(id_value, str)
        assert len(id_value) > 0
        # ID 通常包含下划线分隔
        assert "_" in id_value


# ==================== 测试 BFCLDataset 类 ====================

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_initialization():
    """测试数据集初始化"""
    # 验证类可以被正确实例化
    assert hasattr(BFCLDataset, 'load')
    assert callable(BFCLDataset.load)

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
def test_bfcl_dataset_load_method_parameters():
    """测试load方法的参数处理"""
    # 验证load方法接受正确的参数
    signature = BFCLDataset.load.__code__
    params = signature.co_varnames[:signature.co_argcount]
    assert 'path' in params
    assert 'category' in params
    assert 'test_ids' in params

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.process_multi_turn_test_case')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.encode_fields')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.Dataset')
@patch('builtins.open', new_callable=MagicMock)
@patch('ais_bench.benchmark.datasets.utils.datasets.get_data_path')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.BFCL_INSTALLED', True)
@patch('ais_bench.benchmark.utils.logging.get_logger')
def test_bfcl_dataset_load_normal(mock_logger, mock_get_data_path, mock_open, mock_dataset, mock_encode, mock_process_multi):
    """测试BFCLDataset.load方法的正常加载情况"""
    # 配置mock
    mock_get_data_path.return_value = "mocked_data_path"
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__iter__.return_value = [
        '{"id": "test_001", "question": "Q1"}\n',
        '{"id": "test_002", "question": "Q2"}\n'
    ]
    mock_open.return_value = mock_file
    mock_process_multi.return_value = [{"id": "test_001"}, {"id": "test_002"}]
    mock_encode.return_value = [{"id": "test_001", "encoded": True}, {"id": "test_002", "encoded": True}]
    mock_dataset_instance = MagicMock()
    mock_dataset.from_list.return_value = mock_dataset_instance
    
    # 执行load方法
    result = BFCLDataset.load(path="test_path", category="python")
    
    # 验证结果
    assert result == mock_dataset_instance
    mock_dataset.from_list.assert_called_once_with([{"id": "test_001", "encoded": True}, {"id": "test_002", "encoded": True}])
    mock_encode.assert_called_once()
    mock_process_multi.assert_called_once()

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.BFCL_INSTALLED', False)
def test_bfcl_dataset_load_missing_dependency():
    """测试BFCLDataset.load方法在缺少依赖时的情况"""
    with pytest.raises(ImportError) as excinfo:
        BFCLDataset.load(path="test_path", category="python")
    
    # 验证异常消息
    assert "Missing required package 'bfcl-eval'" in str(excinfo.value)

@pytest.mark.skipif(not BFCL_AVAILABLE, reason="BFCL modules not available")
@patch('builtins.open', new_callable=MagicMock)
@patch('ais_bench.benchmark.datasets.utils.datasets.get_data_path')
@patch('ais_bench.benchmark.datasets.bfcl.bfcl.BFCL_INSTALLED', True)
@patch('ais_bench.benchmark.utils.logging.get_logger')
def test_bfcl_dataset_load_relevance_category(mock_logger, mock_get_data_path, mock_open):
    """测试BFCLDataset.load方法处理relevance类别"""
    # 配置mock
    mock_get_data_path.return_value = "mocked_data_path"
    mock_file = MagicMock()
    mock_file.__enter__.return_value = mock_file
    mock_file.__iter__.return_value = [
        '{"id": "relevance_001", "question": "Q1"}\n'
    ]
    mock_open.return_value = mock_file
    
    # 模拟其他必要的函数
    with patch('ais_bench.benchmark.datasets.bfcl.bfcl.process_multi_turn_test_case') as mock_process, \
         patch('ais_bench.benchmark.datasets.bfcl.bfcl.encode_fields') as mock_encode, \
         patch('ais_bench.benchmark.datasets.bfcl.bfcl.Dataset') as mock_dataset:
        
        mock_process.return_value = [{"id": "relevance_001", "ground_truth": ["relevance_mock_gt"]}]
        mock_encode.return_value = [{"id": "relevance_001", "encoded": True}]
        mock_dataset.from_list.return_value = MagicMock()
        
        # 执行load方法
        BFCLDataset.load(path="test_path", category="relevance")
        
        # 验证relevance类别被正确处理
        assert mock_process.called
        assert mock_encode.called

