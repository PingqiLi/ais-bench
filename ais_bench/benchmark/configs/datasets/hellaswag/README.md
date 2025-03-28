# HellaSwag
## 数据集简介
HellaSwag是一个用于评估自然语言理解能力的基准数据集，主要用于测试模型在常识推理方面的表现。数据集包含多个选择题，要求模型从多个选项中选择最合理的答案。

## 数据集原始获取链接
[https://huggingface.co/datasets/Rowan/hellaswag](https://huggingface.co/datasets/Rowan/hellaswag)

## 数据集内容格式
### 文件结构
```
hellaswag
├── hellaswag.jsonl
├── hellaswag_train_sampled25.jsonl
└── hellaswag_val_contamination_annotations.json
```
### 数据集内容样例格式
|query|choices|gold|
| ----- | ---- | ---- |
|Roof shingle removal: A man is sitting on a roof. He|["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."]|3|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/hellaswag.zip)

## 可用数据集任务
### hellaswag_gen_0_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|hellaswag_gen_0_shot_chat_prompt|hellaswag数据集生成式任务|accuracy|0-shot|对话格式|[hellaswag_gen_0_shot_chat_prompt.py](hellaswag_gen_0_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets hellaswag_gen_0_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.hellaswag.hellaswag_gen_0_shot_chat_prompt import hellaswag_datasets
datasets = [
    *hellaswag_datasets,
]
```

### hellaswag_gen_10_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|hellaswag_gen_10_shot_chat_prompt|hellaswag数据集生成式任务|accuracy|10-shot|对话格式|[hellaswag_gen_10_shot_chat_prompt.py](hellaswag_gen_10_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets hellaswag_gen_10_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.hellaswag.hellaswag_gen_10_shot_chat_prompt import hellaswag_datasets
datasets = [
    *hellaswag_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释

