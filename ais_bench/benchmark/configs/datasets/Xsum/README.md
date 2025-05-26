# Xsum
## 数据集简介
XSum（Extreme Summarization）数据集是用于评估抽象单文档摘要系统的数据集。其目标是创建一个简短的、一句话的新摘要，回答“这篇文章是关于什么的？”这个问题。该数据集包含226711篇新闻文章，每篇文章都附有一句话摘要。这些文章来自BBC（2010年至2017年），涵盖了广泛的领域，如新闻、政治、体育、天气、商业、技术、科学、健康、家庭、教育、娱乐和艺术。
## 数据集原始获取链接
[https://huggingface.co/datasets/EdinburghNLP/xsum](https://huggingface.co/datasets/EdinburghNLP/xsum)
## 数据集内容格式(处理后)
### 文件结构
```
Xsum/
├── dev.csv
├── dev.json
├── dev.jsonl
```
### 数据集内容样例格式

|id|dialogue|summary|
| ---- | ---- | ---- |
|38295789|The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation - a charity to raise money for Nigerian sport.\nMr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42.\nAppearing at the Old Bailey earlier, all four denied the offence.\nThe charge relates to offences which allegedly took place between 2008 and 2014.\nSam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, are due to stand trial in July.\nThey were all released on bail.|Former Premier League footballer Sam Sodje has appeared in court alongside three brothers accused of charity fraud.|

### 处理后的数据集获取链接
[https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
其中`data/Xsum/`就是处理好的数据集

## 可用数据集任务

### Xsum_gen_0_shot_chat.py
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|Xsum_gen_0_shot_chat|Xsum数据集生成式任务|accuracy|0-shot|对话格式|[Xsum_gen_0_shot_chat.py](Xsum_gen_0_shot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets Xsum_gen_0_shot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.Xsum.Xsum_gen_0_shot_chat import Xsum_datasets
datasets = [
    *Xsum_datasets,
]
```

### Xsum_gen_0_shot_str.py
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|Xsum_gen_0_shot_str|Xsum数据集生成式任务|accuracy|0-shot|字符串格式|[Xsum_gen_0_shot_str.py](Xsum_gen_0_shot_str.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets Xsum_gen_0_shot_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.Xsum.Xsum_gen_0_shot_str import Xsum_datasets
datasets = [
    *Xsum_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释