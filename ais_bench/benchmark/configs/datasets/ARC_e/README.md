# ARC Easy Set
## 数据集简介
ARC是一个包含7,787道真实小学阶段科学选择题的新数据集，旨在推动高级问答技术的研究。该数据集分为挑战集（Challenge Set）和简单集（Easy Set），其中挑战集仅包含基于检索算法和词语共现算法均回答错误的难题。本文涉及的是Easy Set。

## 数据集原始获取链接
[https://huggingface.co/datasets/allenai/ai2_arc](https://huggingface.co/datasets/allenai/ai2_arc)

## 数据集内容格式(处理后)
### 文件结构
```
ARC/
└── ARC-e
    ├── ARC-Easy-Dev.jsonl
    └── ARC-Easy-Test.jsonl
```
### 数据集内容样例格式
|id|question|answerKey|
| ----- | ---- | --- |
|Mercury_417466|{"stem":"Which statement best explains why photosynthesis is the foundation of most food webs?","choices":[{"text":"Sunlight is the source of energy for nearly all ecosystems.","label":"A"},{"text":"Most ecosystems are found on land instead of in water.","label":"B"},{"text":"Carbon dioxide is more available than other gases.","label":"C"},{"text":"The producers in all ecosystems are plants.","label":"D"}]}|A|

### 处理后的数据集获取链接
[https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
将其中`data/ARC/`就是处理好的数据集

## 可用数据集任务
### ARC_e_gen_0_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ARC_e_gen_0_shot_chat_prompt|ARC Easy Set数据集生成式任务|accuracy|0-shot|对话格式|[ARC_e_gen_0_shot_chat_prompt.py](ARC_e_gen_0_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets ARC_e_gen_0_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.ARC_e.ARC_e_gen_0_shot_chat_prompt import ARC_e_datasets
datasets = [
    *ARC_e_datasets,
]
```

### ARC_e_gen_25_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ARC_e_gen_25_shot_chat_prompt|ARC Easy Set数据集生成式任务|accuracy|25-shot|对话格式|[ARC_e_gen_25_shot_chat_prompt.py](ARC_e_gen_25_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets ARC_e_gen_25_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.ARC_e.ARC_e_gen_25_shot_chat_prompt import ARC_e_datasets
datasets = [
    *ARC_e_datasets,
]

**注:** 数据集任务的详细配置的含义请参见配置文件的注释