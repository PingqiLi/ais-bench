# AIME2025
## 数据集简介
AIME2025 数据集来源于 2025 年的 American Invitational Mathematics Examination（AIME），包含了一系列中学阶段的高难度数学竞赛题。该考试专为美国高中生设计，旨在筛选进入美国数学奥林匹克（USAMO）的候选人。AIME2025 数据集共收录 30 道正式比赛题，涵盖代数、数论、组合数学、几何等多个方向。每道题都具有单一整数解，题目设计注重逻辑推理和数学建模能力，难度远高于常规中学数学题。适合用于评估模型在复杂数学推理和符号计算方面的能力。

## 数据集原始获取链接
https://huggingface.co/datasets/opencompass/AIME2025

## 数据集内容格式(处理后)
### 文件结构
```
aime2025/
└── aime2025.jsonl
```
### 数据集内容样例格式
|question|answer|
| ---- | ---- |
|Find the sum of all integer bases $b>9$ for which $17_{b}$ is a divisor of $97_{b}$.|70|
### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime2025.zip)


## 可用数据集任务

### aime2025_gen_0_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|aime2025_gen|AIME2025|数据集生成式任务|准确率(accuracy)|0-shot|string|aime2025_gen_0_shot_chat_prompt.py|


#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets aime2025_gen_0_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.aime2025.aime2025_gen_0_shot_chat_prompt import aime_datasets
datasets = [
    *aime_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释