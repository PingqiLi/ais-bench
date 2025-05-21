# LCSTS
## 数据集简介
LCSTS数据集是一个大规模的中文短文本摘要数据集，由哈尔滨工业大学深圳研究生院发布。该数据集主要来源于中国的微博平台，包含了超过200万条真实的中文短文本及其作者给出的简短摘要。此外，研究者还手动标注了其中10666条摘要与对应短文本的相关性。
## 数据集原始获取链接
[https://huggingface.co/datasets/aligeniewcp22/LCSTS](https://huggingface.co/datasets/aligeniewcp22/LCSTS)
## 数据集内容格式(处理后)
### 文件结构
```
LCSTS/
├── test.src.txt
├── test.tgt.txt
```
### 数据集内容样例格式

数据集中每行是一段文本，内容示例如下：
```
63岁退休教师谢淑华，拉着人力板车，历时1年，走了2万4千里路，带着年过九旬的妈妈环游中国，完成了妈妈“一辈子在锅台边转，也想出去走走”的心愿。她说：“妈妈愿意出去走走，我就愿意拉着，孝心不能等，能走多远就走多远。
```

### 处理后的数据集获取链接
[https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
其中`data/LCSTS/`就是处理好的数据集

## 可用数据集任务

### lcsts_gen_0_shot_chat.py
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|lcsts_gen_0_shot_chat|lcsts数据集生成式任务|accuracy|0-shot|对话格式|[lcsts_gen_0_shot_chat.py](lcsts_gen_0_shot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets lcsts_gen_0_shot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.lcsts.lcsts_gen_0_shot_chat import lcsts_datasets
datasets = [
    *lcsts_datasets,
]
```

### lcsts_gen_0_shot_str.py
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|lcsts_gen_0_shot_str|lcsts数据集生成式任务|accuracy|0-shot|字符串格式|[lcsts_gen_0_shot_str.py](lcsts_gen_0_shot_str.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets lcsts_gen_0_shot_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.lcsts.lcsts_gen_0_shot_str import lcsts_datasets
datasets = [
    *lcsts_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释