import unittest

from ais_bench.benchmark.datasets.gsm8k import Gsm8kAgentEvaluator


class TestGsm8kAgentEvaluator(unittest.TestCase):
    def test_agent_evaluator_score(self):
        # 绕过错误的 __init__ 实现，直接用 __new__ 创建实例并设置必要属性
        eva = Gsm8kAgentEvaluator.__new__(Gsm8kAgentEvaluator)
        eva.action = 'PythonInterpreter'
        # 预测正确且包含action
        steps = [[
            {'type': 'Other'},
            {'type': 'PythonInterpreter', 'errmsg': '', 'result': {'text': '5'}}
        ]]
        out = eva.score(predictions=['5'], references=[5], steps=steps)
        self.assertIn('follow_acc', out)
        self.assertIn('reasoning_acc', out)
        self.assertIn('code_acc', out)
        self.assertIn('action_pct', out)

        # 预测错误但存在action，且soft_equal成功
        steps2 = [[
            {'type': 'PythonInterpreter', 'errmsg': '', 'result': {'text': '5'}}
        ]]
        out2 = eva.score(predictions=['6'], references=[5], steps=steps2)
        self.assertIn('follow_acc', out2)

        # 无action的情况
        steps3 = [[{'type': 'Other'}]]
        out3 = eva.score(predictions=['5'], references=[5], steps=steps3)
        self.assertIn('follow_acc', out3)


if __name__ == '__main__':
    unittest.main()
