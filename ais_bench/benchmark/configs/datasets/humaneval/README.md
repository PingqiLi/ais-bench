# HumanEval
## 数据集简介
OpenAI 发布的 HumanEval 数据集包含 164 个编程问题，每个问题都提供了函数签名、文档字符串、函数主体以及多个单元测试。这些问题均为手工编写，以确保它们不会出现在代码生成模型的训练集中。

## 数据集原始获取链接
[https://huggingface.co/datasets/openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval)

## 数据集内容格式(处理后)
### 文件结构
```
humaneval
└── human-eval-v2-20210705.jsonl
```
### 数据集内容样例格式
|task_id|prompt|test|
| ----- | ---- | ---- |
|HumanEval/0|from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    \"\"\" Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    \"\"\"\n", "entry_point": "has_close_elements", "canonical_solution": "    for idx, elem in enumerate(numbers):\n        for idx2, elem2 in enumerate(numbers):\n            if idx != idx2:\n                distance = abs(elem - elem2)\n                if distance < threshold:\n                    return True\n\n    return False\n|\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip)

## 可用数据集任务
### humaneval_gen_0_shot
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|humaneval_gen_0_shot|humaneval数据集生成式任务|pass@1|0-shot|字符串格式|[humaneval_gen_0_shot.py](humaneval_gen_0_shot.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets humaneval_gen_0_shot
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.humaneval.humaneval_gen_0_shot import humaneval_datasets
datasets = [
    *humaneval_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释