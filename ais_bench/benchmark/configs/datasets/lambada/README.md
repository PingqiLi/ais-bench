# LAMBADA
## 数据集简介
LAMBADA（LAnguage Modeling Broadened to Account for Discourse Aspects）数据集是一种开放式填空任务，旨在评估计算模型对文本理解的能力。该数据集包含约10000个从BooksCorpus中提取的段落，每个段落的最后一句话缺少一个目标词，要求模型预测这个缺失的词。
## 数据集原始获取链接
[https://huggingface.co/datasets/cimec/lambada](https://huggingface.co/datasets/cimec/lambada)
## 数据集内容格式(处理后)
### 文件结构
```
lambada/
├── test.jsonl
```
### 数据集内容样例格式
|prompt|label|
| ---- | ---- |
|In my palm is a clear stone, and inside it is a small ivory statuette. A guardian angel.\n\n\"Figured if you're going to be out at night getting hit by cars, you might as well have some backup.\"\n\nI look at him, feeling stunned. Like this is some sort of sign. But as I stare at Harlin, his mouth curved in a confident grin, I don't care about|signs|

### 处理后的数据集获取链接
[https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
其中`data/lambada/`就是处理好的数据集

## 可用数据集任务

### lambada_gen_0_shot_chat.py
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|lambada_gen_0_shot_chat|lambada数据集生成式任务|accuracy|0-shot|对话格式|[lambada_gen_0_shot_chat.py](lambada_gen_0_shot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets lambada_gen_0_shot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.lambada.lambada_gen_0_shot_chat import lambada_datasets
datasets = [
    *lambada_datasets,
]
```

### lambada_gen_0_shot_str.py
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|lambada_gen_0_shot_str|lambada数据集生成式任务|accuracy|0-shot|字符串格式|[lambada_gen_0_shot_str.py](lambada_gen_0_shot_str.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets lambada_gen_0_shot_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.lambada.lambada_gen_0_shot_str import lambada_datasets
datasets = [
    *lambada_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释