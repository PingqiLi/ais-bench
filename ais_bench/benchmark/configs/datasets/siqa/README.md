# SIQA
## 数据集简介
SIQA（Social Interaction QA） 是一个用于测试社会常识智能的问答基准。与许多关注物理或分类知识的先前基准不同，SIQA专注于推理人们的行为及其社会影响。例如，给定一个动作如“杰西看了一场音乐会”和一个问题如“杰西为什么这么做？”，人类可以轻松推断出杰西想“看他最喜欢的表演者”或“享受音乐”，而不是“看看里面发生了什么”或“看看是否有效”。
## 数据集原始获取链接
[https://huggingface.co/datasets/allenai/social_i_qa](https://huggingface.co/datasets/allenai/social_i_qa)
## 数据集内容格式(处理后)
### 文件结构
```
siqa/
├── dev.jsonl
├── dev-labels.lst
├── train.jsonl
├── train-labels.lst
```
### 数据集内容样例格式

|context|question|answerA|answerB|answerC|
| ---- | ---- | ---- | ---- | ---- |
|Tracy didn't go home that evening and resisted Riley's attacks.|What does Tracy need to do before this?|make a new plan|Go home and see Riley|Find somewhere to go|

### 处理后的数据集获取链接
[https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
其中`data/siqa/`就是处理好的数据集

## 可用数据集任务

### siqa_gen_0_shot_chat.py
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|siqa_gen_0_shot_chat|siqa数据集生成式任务|accuracy|0-shot|对话格式|[siqa_gen_0_shot_chat.py](siqa_gen_0_shot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets siqa_gen_0_shot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.siqa.siqa_gen_0_shot_chat import siqa_datasets
datasets = [
    *siqa_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释