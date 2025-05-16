# WinoGrande
## 数据集简介
WinoGrande是一个包含44,000道题目的新型数据集，其设计灵感源自Winograd Schema Challenge（Levesque、Davis和Morgenstern，2011年），但通过调整规模并增强对数据集特定偏见的鲁棒性进行了改进。该任务采用二选一的填空形式，目标是为给定句子选择符合常识推理的正确选项。

## 数据集原始获取链接
[https://huggingface.co/datasets/allenai/winogrande](https://huggingface.co/datasets/allenai/winogrande)

## 数据集内容格式
### 文件结构
```
winogrande
├── dev.jsonl
├── dev-labels.lst
├── eval.py
├── README.md
├── sample-submission-labels.lst
├── test.jsonl
├── train_debiased.jsonl
├── train_debiased-labels.lst
├── train_l.jsonl
├── train_l-labels.lst
├── train_m.jsonl
├── train_m-labels.lst
├── train_s.jsonl
├── train_s-labels.lst
├── train_xl.jsonl
├── train_xl-labels.lst
├── train_xs.jsonl
└── train_xs-labels.lst
```
### 数据集内容样例格式
|qID|sentence|option1|option2|answer|
| ----- | ---- | ---- | --- | --- |
|3FCO4VKOZ4BJQ6IFC0VAIBK4KTWE7U-2|Sarah was a much better surgeon than Maria so _ always got the easier cases.|Sarah|Maria|2|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/winogrande.zip)

## 可用数据集任务
### winogrande_gen_0_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|winogrande_gen_0_shot_chat_prompt|winogrande数据集生成式任务|accuracy|0-shot|对话格式|[winogrande_gen_0_shot_chat_prompt.py](winogrande_gen_0_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets winogrande_gen_0_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_0_shot_chat_prompt import winogrande_datasets
datasets = [
    *winogrande_datasets,
]
```

### winogrande_gen_5_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|winogrande_gen_5_shot_chat_prompt|piqa数据集生成式任务|accuracy|5-shot|对话格式|[winogrande_gen_5_shot_chat_prompt.py](winogrande_gen_5_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets winogrande_gen_5_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.winogrande.winogrande_gen_5_shot_chat_prompt import winogrande_datasets
datasets = [
    *winogrande_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释

