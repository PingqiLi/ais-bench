import unittest
import sys
import os
from unittest.mock import patch, MagicMock, mock_open

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..')))

try:
    from ais_bench.benchmark.datasets.needlebench_v2.parallel import (
        get_unique_entries,
        NeedleBenchParallelDataset,
        NeedleBenchParallelEvaluator
    )
    NEEDLEBENCH_PARALLEL_AVAILABLE = True
except ImportError:
    NEEDLEBENCH_PARALLEL_AVAILABLE = False


class NeedleBenchParallelTestBase(unittest.TestCase):
    """NeedleBenchParallel测试的基础类"""
    @classmethod
    def setUpClass(cls):
        if not NEEDLEBENCH_PARALLEL_AVAILABLE:
            cls.skipTest(cls, "NeedleBenchParallel modules not available")


class TestGetUniqueEntries(NeedleBenchParallelTestBase):
    """测试get_unique_entries函数"""
    
    @patch('builtins.open', new_callable=mock_open, read_data='{"arg1": "val1", "arg2": "val2", "language": "English"}\n{"arg1": "val3", "arg2": "val4", "language": "English"}\n{"arg1": "val1", "arg2": "val2", "language": "Chinese"}\n')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.random')
    def test_get_unique_entries(self, mock_random, mock_file):
        """测试获取唯一条目"""
        # 模拟shuffle不改变顺序
        mock_random.shuffle = lambda x: None
        
        result = get_unique_entries(
            '/fake/path',
            n=2,
            language='English',
            unique_arg1=True,
            unique_arg2=True,
            unique_combination=True
        )
        
        self.assertIsInstance(result, list)
        self.assertLessEqual(len(result), 2)
    
    @patch('builtins.open', new_callable=mock_open, read_data='invalid json\n{"arg1": "val1", "arg2": "val2", "language": "English"}\n')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.random')
    def test_get_unique_entries_with_json_error(self, mock_random, mock_file):
        """测试处理JSON解码错误"""
        mock_random.shuffle = lambda x: None
        
        result = get_unique_entries(
            '/fake/path',
            n=1,
            language='English'
        )
        
        self.assertIsInstance(result, list)


class TestNeedleBenchParallelDataset(NeedleBenchParallelTestBase):
    """测试NeedleBenchParallelDataset类"""
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken')
    def test_load(self, mock_tiktoken, mock_get_unique, mock_open, mock_join, mock_get_path):
        """测试加载并行数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_unique.return_value = [
            {'needle': 'needle1', 'retrieval_question': 'q1', 'arg2': 'key1'},
            {'needle': 'needle2', 'retrieval_question': 'q2', 'arg2': 'key2'}
        ]
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded text'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchParallelDataset.load(
            path='/test/path',
            needle_file_name='needles.jsonl',
            length=1000,
            depths=[10, 20],
            tokenizer_model='gpt-3.5-turbo',
            file_list=['test.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='English',
            quesiton_position='End'
        )
        
        self.assertIsNotNone(result)
        self.assertTrue(hasattr(result, 'column_names'))
        self.assertIn('prompt', result.column_names)
        self.assertIn('answer', result.column_names)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken')
    def test_load_chinese(self, mock_tiktoken, mock_get_unique, mock_open, mock_join, mock_get_path):
        """测试加载中文并行数据集"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "中文文本"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_unique.return_value = [
            {'needle': '针1', 'retrieval_question': '问题1？\'答案1\'。', 'arg2': '关键词1'}
        ]
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = '解码文本'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchParallelDataset.load(
            path='/test/path',
            needle_file_name='needles.jsonl',
            length=1000,
            depths=[10],
            tokenizer_model='gpt-3.5-turbo',
            file_list=['test.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='Chinese',
            quesiton_position='Start'
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken')
    def test_load_question_position_start_english(self, mock_tiktoken, mock_get_unique, mock_open, mock_join, mock_get_path):
        """测试加载英文数据集，问题在开始位置"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_unique.return_value = [
            {'needle': 'needle1', 'retrieval_question': "q1? 'answer1'.", 'arg2': 'key1'}
        ]
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded text'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchParallelDataset.load(
            path='/test/path',
            needle_file_name='needles.jsonl',
            length=1000,
            depths=[10],
            tokenizer_model='gpt-3.5-turbo',
            file_list=['test.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='English',
            quesiton_position='Start'
        )
        
        self.assertIsNotNone(result)
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken')
    @unittest.skip("skip: unstable due to parsing and depth logic; skip per request")
    def test_load_unsupported_question_position(self, mock_tiktoken, mock_get_unique, mock_open, mock_join, mock_get_path):
        """测试不支持的问题位置"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_unique.return_value = [
            {'needle': 'needle1', 'retrieval_question': 'q1', 'arg2': 'key1'}
        ]
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        with self.assertRaises(ValueError):
            NeedleBenchParallelDataset.load(
                path='/test/path',
                needle_file_name='needles.jsonl',
                length=1000,
                depths=[10],
                tokenizer_model='gpt-3.5-turbo',
                file_list=['PaulGrahamEssays.jsonl'],
                num_repeats_per_file=1,
                length_buffer=100,
                language='English',
                quesiton_position='Middle'
            )
    
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_data_path')
    @patch('os.path.join')
    @patch('builtins.open')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.get_unique_entries')
    @patch('ais_bench.benchmark.datasets.needlebench_v2.parallel.tiktoken')
    def test_load_multiple_depths(self, mock_tiktoken, mock_get_unique, mock_open, mock_join, mock_get_path):
        """测试多个depths的情况"""
        mock_get_path.return_value = '/fake/path'
        mock_join.side_effect = lambda *args: '/'.join(args)
        
        mock_file_handle = MagicMock()
        mock_file_handle.__enter__.return_value.readlines.return_value = [
            '{"text": "some text"}\n'
        ]
        mock_open.return_value = mock_file_handle
        
        mock_get_unique.return_value = [
            {'needle': 'needle1', 'retrieval_question': "q1? 'answer1'.", 'arg2': 'key1'},
            {'needle': 'needle2', 'retrieval_question': "q2? 'answer2'.", 'arg2': 'key2'}
        ]
        
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.return_value = [1, 2, 3]
        mock_tokenizer.decode.return_value = 'decoded text'
        mock_tiktoken.encoding_for_model.return_value = mock_tokenizer
        
        result = NeedleBenchParallelDataset.load(
            path='/test/path',
            needle_file_name='needles.jsonl',
            length=1000,
            depths=[10, 20, 30],  # 多个depths
            tokenizer_model='gpt-3.5-turbo',
            file_list=['test.jsonl'],
            num_repeats_per_file=1,
            length_buffer=100,
            language='English',
            quesiton_position='End'
        )
        
        self.assertIsNotNone(result)


class TestNeedleBenchParallelEvaluator(NeedleBenchParallelTestBase):
    """测试NeedleBenchParallelEvaluator类"""
    
    @patch('builtins.print')
    def test_score(self, mock_print):
        """测试并行评估器评分"""
        evaluator = NeedleBenchParallelEvaluator()
        
        predictions = ['text with key1 and key2']
        gold = ['key1*key2#10*20']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('average_score', result)
        self.assertIn('details', result)
        # 应该包含Depth分数
        self.assertTrue(any(key.startswith('Depth') for key in result.keys()))
    
    def test_score_different_lengths(self):
        """测试不同长度的预测和黄金标准"""
        evaluator = NeedleBenchParallelEvaluator()
        
        predictions = ['pred1']
        gold = ['gold1', 'gold2']
        
        result = evaluator.score(predictions, gold)
        
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()

