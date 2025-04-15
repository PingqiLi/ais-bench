# MMLU-Pro
## 数据集简介
MMLU-Pro 数据集是一个更为稳健且具有挑战性的大规模多任务理解数据集，专为更严格地评估大语言模型的能力而设计。该数据集包含了来自多个学科的 12,000 个复杂问题。

## 数据集原始获取链接
[https://huggingface.co/datasets/datasets/TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/datasets/TIGER-Lab/MMLU-Pro)

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
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mmlu_pro.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/humaneval.zip)

## 可用数据集任务
### mmlu_pro_gen_0_shot_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mmlu_pro_gen_0_shot_str|mmlu-pro数据集生成式任务|pass@1|0-shot|字符串格式|[mmlu_pro_gen_0_shot_str.py](mmlu_pro_gen_0_shot_str.py)|

#### 命令行调用
```shell
# 每个数据集文件作为单一task来评测
## 字符串格式后端
ais_bench --models vllm_api_general --datasets mmlu_pro_gen_0_shot_str
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets mmlu_pro_gen_0_shot_str

# 合并多个数据集文件统一评测
## 字符串格式后端
ais_bench --models vllm_api_general --datasets mmlu_pro_gen_0_shot_str --merge-ds
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets mmlu_pro_gen_0_shot_str --merge-ds

```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.mmlu_pro.mmlu_pro_gen_0_shot_str import mmlu_pro_datasets
datasets = [
    *mmlu_pro_datasets,
]
```

### mmlu_pro_gen_5_shot_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mmlu_pro_gen_5_shot_str|mmlu-pro数据集生成式任务|pass@1|0-shot|字符串格式|[mmlu_pro_gen_5_shot_str.py](mmlu_pro_gen_5_shot_str.py)|

#### 命令行调用
```shell
# 每个数据集文件作为单一task来评测
## 字符串格式后端
ais_bench --models vllm_api_general --datasets mmlu_pro_gen_5_shot_str
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets mmlu_pro_gen_5_shot_str

# 合并多个数据集文件统一评测
## 字符串格式后端
ais_bench --models vllm_api_general --datasets mmlu_pro_gen_5_shot_str --merge-ds
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets mmlu_pro_gen_5_shot_str --merge-ds
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.mmlu_pro.mmlu_pro_gen_5_shot_str import mmlu_pro_datasets
datasets = [
    *mmlu_pro_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释