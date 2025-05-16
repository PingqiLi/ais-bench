# IFEval
## 数据集简介
IFEval是一个用于评估大语言模型（如GPT-4、PaLM 2等）指令遵循能力的数据集。随着大语言模型在自然语言任务中的广泛应用，模型的指令遵循能力成为一个重要的评估指标。

**注意**：数据集运行前请先安装依赖[extra.txt](../../../../../requirements/extra.txt)
```shell
# 需要处在最外层benchmark文件夹下，运行下列指令：
pip3 install -r requirements/extra.txt
```

## 数据集原始获取链接
[https://huggingface.co/datasets/google/IFEval](https://huggingface.co/datasets/google/IFEval)

## 数据集内容格式(处理后)
### 文件结构
```
ifeval
└── input_data.jsonl
```

### 数据集内容样例格式
|key|prompt|instruction_id_list|kwargs|
| ----- | ---- | ---- | ---- |
|1000|Write a 300+ word summary of the wikipedia page \"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*.|["punctuation:no_comma", "detectable_format:number_highlighted_sections", "length_constraints:number_words"]|[{}, {"num_highlights": 3}, {"relation": "at least", "num_words": 300}]|


### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/ifeval.zip)

## 可用数据集任务
### ifeval_0_shot_gen_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|ifeval_0_shot_gen_str|ifeval数据集生成式任务|accuracy|0-shot|字符串格式|[ifeval_0_shot_gen_str.py](ifeval_0_shot_gen_str.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets ifeval_0_shot_gen_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.ifeval.ifeval_0_shot_gen_str import ifeval_datasets
datasets = [
    *ifeval_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释