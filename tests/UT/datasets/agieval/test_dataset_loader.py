"""Unit tests for agieval/dataset_loader.py"""
import json
import unittest
from unittest.mock import patch, mock_open, MagicMock

import pandas as pd

from ais_bench.benchmark.datasets.agieval.dataset_loader import (
    convert_zero_shot,
    convert_zero_shot_CoT_stage1,
    combine_prompt,
    concat_prompt,
    concat_prompt_chat_mode,
    convert_few_shot,
    load_dataset,
    generate_second_stage_input,
    load_dataset_as_result_schema,
    english_qa_datasets,
    chinese_qa_datasets,
    english_cloze_datasets,
    chinese_cloze_datasets,
)


class TestConvertZeroShot(unittest.TestCase):
    """测试 convert_zero_shot"""

    def test_english_qa(self):
        """测试英文 QA"""
        line = {
            'passage': 'Test passage',
            'question': 'What is the answer?',
            'options': ['A. Option 1', 'B. Option 2', 'C. Option 3']
        }
        result = convert_zero_shot(line, 'lsat-ar')
        self.assertIn('Test passage', result)
        self.assertIn('What is the answer?', result)
        self.assertIn('Answer Choices:', result)
        self.assertIn('Among A through C', result)

    def test_chinese_qa(self):
        """测试中文 QA"""
        line = {
            'passage': '测试段落',
            'question': '答案是什么？',
            'options': ['A. 选项1', 'B. 选项2']
        }
        result = convert_zero_shot(line, 'logiqa-zh')
        self.assertIn('测试段落', result)
        self.assertIn('答案是什么？', result)
        self.assertIn('选项：', result)
        self.assertIn('从A到B', result)

    def test_english_cloze(self):
        """测试英文完形填空"""
        line = {
            'passage': 'Math problem',
            'question': 'Solve x + 1 = 2'
        }
        result = convert_zero_shot(line, 'math')
        self.assertIn('Math problem', result)
        self.assertIn('Solve x + 1 = 2', result)
        self.assertIn('The answer is', result)

    def test_chinese_cloze(self):
        """测试中文完形填空"""
        line = {
            'passage': '数学问题',
            'question': '求解 x + 1 = 2'
        }
        result = convert_zero_shot(line, 'gaokao-mathcloze')
        self.assertIn('数学问题', result)
        self.assertIn('求解 x + 1 = 2', result)
        self.assertIn('答案：', result)

    def test_none_passage(self):
        """测试 passage 为 None"""
        line = {
            'passage': None,
            'question': 'Test question',
            'options': ['A. Opt1']
        }
        result = convert_zero_shot(line, 'lsat-ar')
        self.assertIn('Test question', result)

    def test_single_option_english(self):
        """测试单个选项（英文）"""
        line = {
            'passage': '',
            'question': 'Test?',
            'options': ['A. Only one']
        }
        result = convert_zero_shot(line, 'lsat-ar')
        self.assertIn('Among A through E', result)  # count 变为 5

    def test_single_option_chinese(self):
        """测试单个选项（中文）"""
        line = {
            'passage': '',
            'question': '测试？',
            'options': ['A. 唯一选项']
        }
        result = convert_zero_shot(line, 'logiqa-zh')
        self.assertIn('从A到D', result)  # count 变为 4

    def test_unknown_dataset(self):
        """测试未知数据集"""
        line = {
            'passage': 'Test',
            'question': 'Question?',
            'options': []
        }
        result = convert_zero_shot(line, 'unknown-dataset')
        self.assertIn('Test', result)
        self.assertIn('Question?', result)

    def test_exception_handling(self):
        """测试异常处理"""
        line = {}
        result = convert_zero_shot(line, 'lsat-ar')
        self.assertIsNotNone(result)


class TestConvertZeroShotCoT(unittest.TestCase):
    """测试 convert_zero_shot_CoT_stage1"""

    def test_english_qa(self):
        """测试英文 QA CoT"""
        line = {
            'passage': 'Passage',
            'question': 'Question?',
            'options': ['A. Opt1', 'B. Opt2']
        }
        result = convert_zero_shot_CoT_stage1(line, 'lsat-ar')
        self.assertIn('Passage', result)
        self.assertIn("Let's think step by step", result)

    def test_chinese_qa(self):
        """测试中文 QA CoT"""
        line = {
            'passage': '段落',
            'question': '问题？',
            'options': ['A. 选项1', 'B. 选项2']
        }
        result = convert_zero_shot_CoT_stage1(line, 'logiqa-zh')
        self.assertIn('段落', result)
        self.assertIn('让我们逐步思考', result)

    def test_english_cloze(self):
        """测试英文完形填空 CoT"""
        line = {
            'passage': 'Math',
            'question': 'Solve'
        }
        result = convert_zero_shot_CoT_stage1(line, 'math')
        self.assertIn("Let's think step by step", result)

    def test_chinese_cloze(self):
        """测试中文完形填空 CoT"""
        line = {
            'passage': '数学',
            'question': '求解'
        }
        result = convert_zero_shot_CoT_stage1(line, 'gaokao-mathcloze')
        self.assertIn('让我们逐步思考', result)


class TestCombinePrompt(unittest.TestCase):
    """测试 combine_prompt"""

    @patch('pandas.read_csv')
    def test_combine_prompt_english(self, mock_read_csv):
        """测试组合提示（英文）"""
        mock_context_df = pd.DataFrame({
            'lsat-ar': [
                "{'passage': 'P1', 'question': 'Q1', 'options': ['A', 'B'], 'label': 'A', 'answer': None}",
                "{'passage': 'P2', 'question': 'Q2', 'options': ['C', 'D'], 'label': 'C', 'answer': None}"
            ]
        })
        mock_explanation_df = pd.DataFrame({
            'lsat-ar': ['Explanation 1', 'Explanation 2']
        })
        mock_read_csv.side_effect = [mock_context_df, mock_explanation_df]
        
        result = combine_prompt('/fake/path.csv', 'lsat-ar', load_explanation=True)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn('Problem 1', result[0])

    @patch('pandas.read_csv')
    def test_combine_prompt_chinese(self, mock_read_csv):
        """测试组合提示（中文）"""
        mock_context_df = pd.DataFrame({
            'logiqa-zh': [
                "{'passage': '段落1', 'question': '问题1', 'options': ['A', 'B'], 'label': 'A', 'answer': None}"
            ]
        })
        mock_explanation_df = pd.DataFrame({
            'logiqa-zh': ['解析1']
        })
        mock_read_csv.side_effect = [mock_context_df, mock_explanation_df]
        
        result = combine_prompt('/fake/path.csv', 'logiqa-zh', load_explanation=True)
        self.assertIn('问题 1', result[0])

    @patch('pandas.read_csv')
    def test_combine_prompt_chat_mode(self, mock_read_csv):
        """测试聊天模式"""
        mock_context_df = pd.DataFrame({
            'lsat-ar': [
                "{'passage': 'P1', 'question': 'Q1', 'options': ['A', 'B'], 'label': 'A', 'answer': None}"
            ]
        })
        mock_explanation_df = pd.DataFrame({
            'lsat-ar': ['Explanation 1']
        })
        mock_read_csv.side_effect = [mock_context_df, mock_explanation_df]
        
        result = combine_prompt('/fake/path.csv', 'lsat-ar', chat_mode=True)
        self.assertIsInstance(result, list)
        self.assertIsInstance(result[0], tuple)

    @patch('pandas.read_csv')
    def test_combine_prompt_without_explanation(self, mock_read_csv):
        """测试不加载解释"""
        mock_context_df = pd.DataFrame({
            'lsat-ar': [
                "{'passage': 'P1', 'question': 'Q1', 'options': ['A', 'B'], 'label': 'A', 'answer': None}"
            ]
        })
        mock_explanation_df = pd.DataFrame({
            'lsat-ar': ['Explanation 1']
        })
        mock_read_csv.side_effect = [mock_context_df, mock_explanation_df]
        
        result = combine_prompt('/fake/path.csv', 'lsat-ar', load_explanation=False)
        self.assertNotIn('Explanation for Problem', result[0])


class TestConcatPrompt(unittest.TestCase):
    """测试 concat_prompt"""

    @patch('tiktoken.encoding_for_model')
    def test_concat_prompt_english(self, mock_tiktoken):
        """测试拼接提示（英文）"""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 100  # 模拟 100 个 token
        mock_tiktoken.return_value = mock_enc
        
        demos = ['Demo 1', 'Demo 2', 'Demo 3']
        result, n_shot = concat_prompt(demos, 'lsat-ar', max_tokens=500)
        self.assertIsInstance(result, str)
        self.assertGreater(n_shot, 0)

    @patch('tiktoken.encoding_for_model')
    def test_concat_prompt_chinese(self, mock_tiktoken):
        """测试拼接提示（中文）"""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 100
        mock_tiktoken.return_value = mock_enc
        
        demos = ['示例 1', '示例 2']
        result, n_shot = concat_prompt(demos, 'logiqa-zh', max_tokens=500)
        self.assertIsInstance(result, str)

    @patch('tiktoken.encoding_for_model')
    def test_concat_prompt_max_tokens_exceeded(self, mock_tiktoken):
        """测试超过最大 token"""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 1000  # 模拟很多 token
        mock_tiktoken.return_value = mock_enc
        
        demos = ['Demo 1', 'Demo 2', 'Demo 3']
        result, n_shot = concat_prompt(demos, 'lsat-ar', max_tokens=50)
        self.assertLessEqual(n_shot, len(demos))


class TestConcatPromptChatMode(unittest.TestCase):
    """测试 concat_prompt_chat_mode"""

    @patch('tiktoken.encoding_for_model')
    def test_concat_prompt_chat_mode(self, mock_tiktoken):
        """测试聊天模式拼接"""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 100
        mock_tiktoken.return_value = mock_enc
        
        demos = [('Question 1', 'Answer 1'), ('Question 2', 'Answer 2')]
        result, n_shot = concat_prompt_chat_mode(demos, 'lsat-ar', max_tokens=500)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertEqual(result[0]['role'], 'user')

    @patch('tiktoken.encoding_for_model')
    def test_concat_prompt_chat_mode_max_tokens(self, mock_tiktoken):
        """测试聊天模式超过最大 token"""
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 1000
        mock_tiktoken.return_value = mock_enc
        
        demos = [('Q1', 'A1'), ('Q2', 'A2'), ('Q3', 'A3')]
        result, n_shot = concat_prompt_chat_mode(demos, 'lsat-ar', max_tokens=50)
        self.assertLessEqual(len(result), len(demos) * 2)


class TestConvertFewShot(unittest.TestCase):
    """测试 convert_few_shot"""

    def test_convert_few_shot_english(self):
        """测试 few-shot 转换（英文）"""
        line = {
            'passage': 'Passage',
            'question': 'Question?',
            'options': ['A. Opt1', 'B. Opt2']
        }
        demo = 'Demo content\n'
        result = convert_few_shot(line, 'lsat-ar', demo, 2)
        self.assertIn('Demo content', result)
        self.assertIn('Problem 3', result)
        self.assertIn('Question?', result)

    def test_convert_few_shot_chinese(self):
        """测试 few-shot 转换（中文）"""
        line = {
            'passage': '段落',
            'question': '问题？',
            'options': ['A. 选项1']
        }
        demo = '示例内容\n'
        result = convert_few_shot(line, 'logiqa-zh', demo, 1)
        self.assertIn('示例内容', result)
        self.assertIn('问题 2', result)

    def test_convert_few_shot_chat_mode(self):
        """测试 few-shot 聊天模式"""
        line = {
            'passage': '',
            'question': 'Q?',
            'options': ['A']
        }
        demo = [{'role': 'user', 'content': 'Demo'}]
        result = convert_few_shot(line, 'lsat-ar', demo, 0, chat_mode=True)
        self.assertIsInstance(result, list)
        self.assertEqual(result[-1]['role'], 'user')


class TestLoadDataset(unittest.TestCase):
    """测试 load_dataset"""

    @patch('ais_bench.benchmark.datasets.agieval.dataset_loader.read_jsonl')
    def test_load_dataset_zero_shot(self, mock_read_jsonl):
        """测试加载数据集（zero-shot）"""
        mock_read_jsonl.return_value = [
            {'passage': 'P', 'question': 'Q?', 'options': ['A', 'B']}
        ]
        
        result = load_dataset('lsat-ar', 'zero-shot', '/fake/path')
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @patch('ais_bench.benchmark.datasets.agieval.dataset_loader.read_jsonl')
    def test_load_dataset_zero_shot_cot(self, mock_read_jsonl):
        """测试加载数据集（zero-shot-CoT）"""
        mock_read_jsonl.return_value = [
            {'passage': 'P', 'question': 'Q?', 'options': ['A']}
        ]
        
        result = load_dataset('lsat-ar', 'zero-shot-CoT', '/fake/path')
        self.assertIsInstance(result, list)

    @patch('tiktoken.encoding_for_model')
    @patch('pandas.read_csv')
    @patch('ais_bench.benchmark.datasets.agieval.dataset_loader.read_jsonl')
    def test_load_dataset_few_shot(self, mock_read_jsonl, mock_read_csv, mock_tiktoken):
        """测试加载数据集（few-shot）"""
        mock_read_jsonl.return_value = [
            {'passage': 'P', 'question': 'Q?', 'options': ['A', 'B']}
        ]
        mock_context_df = pd.DataFrame({
            'lsat-ar': [
                "{'passage': 'P1', 'question': 'Q1', 'options': ['A', 'B'], 'label': 'A', 'answer': None}"
            ]
        })
        mock_explanation_df = pd.DataFrame({
            'lsat-ar': ['Explanation']
        })
        mock_read_csv.side_effect = [mock_context_df, mock_explanation_df]
        
        mock_enc = MagicMock()
        mock_enc.encode.return_value = [1] * 100
        mock_tiktoken.return_value = mock_enc
        
        result = load_dataset('lsat-ar', 'few-shot', '/fake/path', 
                            prompt_path='/fake/prompt.csv', max_tokens=500)
        self.assertIsInstance(result, list)

    @patch('ais_bench.benchmark.datasets.agieval.dataset_loader.read_jsonl')
    def test_load_dataset_modelscope(self, mock_read_jsonl):
        """测试 ModelScope 数据源"""
        mock_ms_dataset = MagicMock()
        mock_ms_dataset.load.return_value = [
            {'passage': 'P', 'question': 'Q?', 'options': ['A']}
        ]
        
        with patch.dict('sys.modules', {'modelscope': MagicMock(MsDataset=mock_ms_dataset)}):
            with patch.dict('os.environ', {'DATASET_SOURCE': 'ModelScope'}):
                result = load_dataset('lsat-ar', 'zero-shot', '/fake/path')
                self.assertIsInstance(result, list)


class TestGenerateSecondStageInput(unittest.TestCase):
    """测试 generate_second_stage_input"""

    def test_generate_second_stage_english(self):
        """测试生成第二阶段输入（英文）"""
        input_list = [{'context': 'Context', 'metadata': 0}]
        output_list = ['Output']
        
        result = generate_second_stage_input('lsat-ar', input_list, output_list)
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)
        self.assertIn('Therefore', result[0]['context'])

    def test_generate_second_stage_chinese(self):
        """测试生成第二阶段输入（中文）"""
        input_list = [{'context': '上下文', 'metadata': 0}]
        output_list = ['输出']
        
        result = generate_second_stage_input('logiqa-zh', input_list, output_list)
        self.assertIn('因此', result[0]['context'])

    def test_generate_second_stage_with_format_prompt(self):
        """测试带格式提示"""
        input_list = [{'context': 'Context', 'metadata': 0}]
        output_list = ['Output']
        
        result = generate_second_stage_input('lsat-ar', input_list, output_list, 
                                            with_format_prompt=True)
        self.assertIn('extract the final answer', result[0]['context'])


class TestLoadDatasetAsResultSchema(unittest.TestCase):
    """测试 load_dataset_as_result_schema"""

    @patch('ais_bench.benchmark.datasets.agieval.dataset_loader.read_jsonl')
    def test_load_dataset_as_result_schema(self, mock_read_jsonl):
        """测试加载为结果模式"""
        mock_read_jsonl.return_value = [
            {'passage': 'P', 'question': 'Q?', 'options': ['A'], 'label': 'A'}
        ]
        
        result = load_dataset_as_result_schema('lsat-ar', '/fake/path')
        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)

    @patch('ais_bench.benchmark.datasets.agieval.dataset_loader.read_jsonl')
    def test_load_dataset_as_result_schema_with_answer(self, mock_read_jsonl):
        """测试加载为结果模式（使用 answer 字段）"""
        mock_read_jsonl.return_value = [
            {'passage': 'P', 'question': 'Q?', 'answer': '42'}
        ]
        
        result = load_dataset_as_result_schema('math', '/fake/path')
        self.assertIsInstance(result, list)


if __name__ == '__main__':
    unittest.main()

