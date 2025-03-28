# piqa
## 数据集简介
多语言小学数学能力测评基准（MGSM）是一个专注于小学数学题目的评估基准。

## 数据集原始获取链接
[https://huggingface.co/datasets/ybisk/piqa](https://huggingface.co/datasets/ybisk/piqa)

## 数据集内容格式
### 文件结构
```
physicaliqa-train-dev
├── dev.jsonl
├── dev-labels.lst
├── train.jsonl
└── train-labels.lst
```
### 数据集内容样例格式
|id|goal|sol1|sol2|
| ----- | ---- | ---- | --- |
|c36c629e-12e9-43cc-8936-e1a96d869ab0|How do I ready a guinea pig cage for it's new occupants?|Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.|Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish.|

### 处理后的数据集获取链接
[https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip](https://storage.googleapis.com/ai2-mosaic/public/physicaliqa/physicaliqa-train-dev.zip)

## 可用数据集任务
### piqa_gen_0_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|piqa_gen_0_shot_chat_prompt|piqa数据集生成式任务|accuracy|0-shot|对话格式|[piqa_gen_0_shot_chat_prompt.py](piqa_gen_0_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets piqa_gen_0_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.piqa.piqa_gen_0_shot_chat_prompt import piqa_datasets
datasets = [
    *piqa_datasets,
]
```

### piqa_gen_0_shot_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|piqa_gen_0_shot_str|piqa数据集生成式任务|accuracy|0-shot|字符串格式|[piqa_gen_0_shot_str.py](piqa_gen_0_shot_str.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets piqa_gen_0_shot_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.piqa.piqa_gen_0_shot_str import piqa_datasets
datasets = [
    *piqa_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释

