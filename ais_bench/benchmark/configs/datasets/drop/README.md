# DROP
## 数据集简介
DROP 是一个通过众包和对抗性创建的、包含 96,000 个问题的基准测试。在该测试中，系统必须解析问题中的引用（可能涉及多个输入位置），并对这些引用执行离散操作（例如加法、计数或排序）。这些操作要求对段落内容的理解比之前的数据集更加全面和深入

## 数据集原始获取链接
[https://huggingface.co/datasets/ucinlp/drop](https://huggingface.co/datasets/ucinlp/drop)

## 数据集内容格式(处理后)
### 文件结构
```
drop_simple_eval
└── dev.jsonl
```
### 数据集内容样例格式
|context|completion|ref_text|
| ----- | ---- | ---- |
|Passage: In the city, the age distribution of the population shows 21.8% under the age of 18, 13.1% from 18 to 24, 31.7% from 25 to 44, 20.1% from 45 to 64, and 13.2% who were 65 years of age or older. The median age was 34 years. For every 100 females, there were 87.1 males. For every 100 females age 18 and over, there were 83.5 males.\nQuestion: Which age groups each made up more than 20% of the population?\nAnswer:| under the age of 18 and 25 to 44 and 45 to 64| under the age of 18 and 25 to 44 and 45 to 64|


### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/drop_simple_eval.zip)

## 可用数据集任务
### drop_gen_a2697c_0shot
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|drop_gen_a2697c_0shot|drop数据集生成式任务, 默认max out tokens长度取32768|accuracy(pass@1)|0-shot|string|[drop_gen_a2697c_0shot.py](drop_gen_a2697c_0shot.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets drop_gen_a2697c_0shot
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.drop.drop_gen_a2697c_0shot import drop_datasets
datasets = [
    *drop_datasets,
]
```

### drop_gen_a2697c_3shot
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|drop_gen_a2697c_3shot|drop数据集生成式任务, 默认max out tokens长度取32768|accuracy(pass@1)|3-shot|string|[drop_gen_a2697c_3shot.py](drop_gen_a2697c_3shot.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets drop_gen_a2697c_3shot
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.drop.drop_gen_a2697c_3shot import drop_datasets
datasets = [
    *drop_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释