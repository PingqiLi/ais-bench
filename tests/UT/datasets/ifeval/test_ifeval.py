import sys
import os
import unittest
from unittest.mock import patch, mock_open, MagicMock

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../../')))

from ais_bench.benchmark.datasets.ifeval.ifeval import IFEvalDataset, IFEvaluator

# 准备测试数据
dataset_test_data = {
    'sample_data': [
        '{"key": 1, "instruction_id_list": ["keywords:existence"], "prompt": "测试提示1", "kwargs": [{"keywords": ["测试"]}]}',
        '{"key": 2, "instruction_id_list": ["length_constraints:number_words"], "prompt": "测试提示2", "kwargs": [{"min_words": 10}]}'
    ],
    'expected_datasets': [
        {'prompt': '测试提示1', 'reference': {'key': 1, 'instruction_id_list': ['keywords:existence'], 'prompt': '测试提示1', 'kwargs': [{'keywords': ['测试']}]}},
        {'prompt': '测试提示2', 'reference': {'key': 2, 'instruction_id_list': ['length_constraints:number_words'], 'prompt': '测试提示2', 'kwargs': [{'min_words': 10}]}}
    ]
}

basic_test_data = {
    'references': [
        {
            'key': 1,
            'instruction_id_list': ['test_instruction_1'],
            'prompt': '测试提示',
            'kwargs': [{'param1': 'value1'}]
        }
    ],
    'origin_prompt': ['测试提示'],
    'predictions': ['测试响应']
}

multiple_instructions_test_data = [
    {
        'key': 1,
        'instruction_id_list': ['test_instruction_1', 'test_instruction_2'],
        'prompt': '测试提示',
        'kwargs': [{'param1': 'value1'}, {'param2': 'value2'}]
    }
]

none_kwargs_test_data = [
    {
        'key': 1,
        'instruction_id_list': ['test_instruction_1'],
        'prompt': '测试提示',
        'kwargs': [{'param1': None, 'param2': 'value2'}]
    }
]

# IFEvalDataset测试类
class TestIFEvalDataset(unittest.TestCase):
    """测试IFEvalDataset类"""
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.get_data_path')
    def test_load_normal(self, mock_get_data_path):
        """测试正常加载数据"""
        # 设置mock返回值
        mock_get_data_path.return_value = '/test/path/data.jsonl'
        
        # 调用load方法
        with patch('builtins.open', mock_open(read_data='\n'.join(dataset_test_data['sample_data']))), \
             patch('ais_bench.benchmark.datasets.ifeval.ifeval.Dataset') as mock_dataset:
            mock_dataset.from_list.return_value = 'test_dataset'
            result = IFEvalDataset.load('/test/path')
            
            # 验证结果
            mock_get_data_path.assert_called_once_with('/test/path', local_mode=True)
            mock_dataset.from_list.assert_called_once_with(dataset_test_data['expected_datasets'])
            self.assertEqual(result, 'test_dataset')
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.get_data_path')
    def test_load_empty_file(self, mock_get_data_path):
        """测试加载空文件"""
        mock_get_data_path.return_value = '/test/path/empty.jsonl'
        
        # 模拟空文件，当迭代时不会产生任何行
        with patch('builtins.open', mock_open(read_data='')) as mock_file, \
             patch('ais_bench.benchmark.datasets.ifeval.ifeval.Dataset') as mock_dataset:
            mock_dataset.from_list.return_value = 'empty_dataset'
            result = IFEvalDataset.load('/test/path/empty.jsonl')
            
            mock_dataset.from_list.assert_called_once_with([])
            self.assertEqual(result, 'empty_dataset')
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.get_data_path')
    @patch('builtins.open')
    def test_load_file_error(self, mock_open_file, mock_get_data_path):
        """测试文件读取错误"""
        mock_get_data_path.return_value = '/test/path/nonexistent.jsonl'
        mock_open_file.side_effect = FileNotFoundError("文件不存在")
        
        with self.assertRaises(FileNotFoundError):
            IFEvalDataset.load('/test/path/nonexistent.jsonl')

# IFEvaluator测试类
class TestIFEvaluator(unittest.TestCase):
    """测试IFEvaluator类"""
    
    def setUp(self):
        """设置测试环境"""
        self.evaluator_instance = IFEvaluator()
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_strict')
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_loose')
    def test_score_all_correct(self, mock_loose_test, mock_strict_test):
        """测试所有指令都正确执行的情况"""
        # 创建mock的OutputExample
        mock_strict_output = MagicMock()
        mock_strict_output.follow_instruction_list = [True]
        mock_strict_output.instruction_id_list = ['test_instruction_1']
        
        mock_loose_output = MagicMock()
        mock_loose_output.follow_instruction_list = [True]
        mock_loose_output.instruction_id_list = ['test_instruction_1']
        
        mock_strict_test.return_value = mock_strict_output
        mock_loose_test.return_value = mock_loose_output
        
        # 调用score方法
        result = self.evaluator_instance.score(
            basic_test_data['predictions'], 
            basic_test_data['references'], 
            basic_test_data['origin_prompt']
        )
        
        # 验证结果
        self.assertEqual(result['Prompt-level-strict-accuracy'], 100.0)
        self.assertEqual(result['Inst-level-strict-accuracy'], 100.0)
        self.assertEqual(result['Prompt-level-loose-accuracy'], 100.0)
        self.assertEqual(result['Inst-level-loose-accuracy'], 100.0)
        self.assertEqual(result['details']['0']['grade'], 'strict')
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_strict')
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_loose')
    def test_score_loose_correct(self, mock_loose_test, mock_strict_test):
        """测试严格模式不正确但宽松模式正确的情况"""
        # 创建mock的OutputExample
        mock_strict_output = MagicMock()
        mock_strict_output.follow_instruction_list = [False]
        mock_strict_output.instruction_id_list = ['test_instruction_1']
        
        mock_loose_output = MagicMock()
        mock_loose_output.follow_instruction_list = [True]
        mock_loose_output.instruction_id_list = ['test_instruction_1']
        
        mock_strict_test.return_value = mock_strict_output
        mock_loose_test.return_value = mock_loose_output
        
        # 调用score方法
        result = self.evaluator_instance.score(
            basic_test_data['predictions'], 
            basic_test_data['references'], 
            basic_test_data['origin_prompt']
        )
        
        # 验证结果
        self.assertEqual(result['Prompt-level-strict-accuracy'], 0.0)
        self.assertEqual(result['Inst-level-strict-accuracy'], 0.0)
        self.assertEqual(result['Prompt-level-loose-accuracy'], 100.0)
        self.assertEqual(result['Inst-level-loose-accuracy'], 100.0)
        self.assertEqual(result['details']['0']['grade'], 'loose')
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_strict')
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_loose')
    def test_score_all_incorrect(self, mock_loose_test, mock_strict_test):
        """测试所有模式都不正确的情况"""
        # 创建mock的OutputExample
        mock_strict_output = MagicMock()
        mock_strict_output.follow_instruction_list = [False]
        mock_strict_output.instruction_id_list = ['test_instruction_1']
        
        mock_loose_output = MagicMock()
        mock_loose_output.follow_instruction_list = [False]
        mock_loose_output.instruction_id_list = ['test_instruction_1']
        
        mock_strict_test.return_value = mock_strict_output
        mock_loose_test.return_value = mock_loose_output
        
        # 调用score方法
        result = self.evaluator_instance.score(
            basic_test_data['predictions'], 
            basic_test_data['references'], 
            basic_test_data['origin_prompt']
        )
        
        # 验证结果
        self.assertEqual(result['Prompt-level-strict-accuracy'], 0.0)
        self.assertEqual(result['Inst-level-strict-accuracy'], 0.0)
        self.assertEqual(result['Prompt-level-loose-accuracy'], 0.0)
        self.assertEqual(result['Inst-level-loose-accuracy'], 0.0)
        self.assertEqual(result['details']['0']['grade'], 'none')
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_strict')
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_loose')
    def test_score_multiple_instructions(self, mock_loose_test, mock_strict_test):
        """测试多个指令的情况"""
        # 创建mock的OutputExample
        mock_strict_output = MagicMock()
        mock_strict_output.follow_instruction_list = [True, False]
        mock_strict_output.instruction_id_list = ['test_instruction_1', 'test_instruction_2']
        
        mock_loose_output = MagicMock()
        mock_loose_output.follow_instruction_list = [True, True]
        mock_loose_output.instruction_id_list = ['test_instruction_1', 'test_instruction_2']
        
        mock_strict_test.return_value = mock_strict_output
        mock_loose_test.return_value = mock_loose_output
        
        # 调用score方法
        result = self.evaluator_instance.score(
            basic_test_data['predictions'], 
            multiple_instructions_test_data, 
            basic_test_data['origin_prompt']
        )
        
        # 验证结果
        self.assertEqual(result['Prompt-level-strict-accuracy'], 0.0)  # 因为不是所有指令都通过
        self.assertEqual(result['Inst-level-strict-accuracy'], 50.0)   # 2个指令中1个通过
        self.assertEqual(result['Prompt-level-loose-accuracy'], 100.0) # 所有指令都通过
        self.assertEqual(result['Inst-level-loose-accuracy'], 100.0)
        self.assertEqual(result['details']['0']['grade'], 'loose')
    
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_strict')
    @patch('ais_bench.benchmark.datasets.ifeval.ifeval.test_instruction_following_loose')
    def test_score_empty_kwargs(self, mock_loose_test, mock_strict_test):
        """测试kwargs中包含None值的情况"""
        # 创建mock的OutputExample
        mock_strict_output = MagicMock()
        mock_strict_output.follow_instruction_list = [True]
        mock_strict_output.instruction_id_list = ['test_instruction_1']
        
        mock_loose_output = MagicMock()
        mock_loose_output.follow_instruction_list = [True]
        mock_loose_output.instruction_id_list = ['test_instruction_1']
        
        mock_strict_test.return_value = mock_strict_output
        mock_loose_test.return_value = mock_loose_output
        
        # 调用score方法
        result = self.evaluator_instance.score(
            basic_test_data['predictions'], 
            none_kwargs_test_data, 
            basic_test_data['origin_prompt']
        )
        
        # 验证结果
        self.assertEqual(result['Prompt-level-strict-accuracy'], 100.0)
        self.assertEqual(result['details']['0']['grade'], 'strict')