# GSM8K
## 数据集简介
GSM8K 数据集由人类出题者编写的 8500 道高质量的小学数学题组成。我们将这些题目划分为 7500 道训练题和 1000 道测试题。这些题目需要 2 到 8 个步骤来求解，解题方法主要是通过运用基本的算术运算（加、减、除、乘）进行一系列的基础计算，从而得出最终答案。一个聪明的中学生应该能够解出每一道题。
## 数据集原始获取链接
[https://github.com/openai/grade-school-math](https://github.com/openai/grade-school-math)
## 数据集内容格式(处理后)
### 文件结构
```
gsm8k/
├── test.jsonl
├── test_socratic.jsonl
├── train.jsonl
└── train_socratic.jsonl
```
### 数据集内容样例格式
|question|answer|
| ---- | ---- |
|Janet\u2019s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?|Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 duck eggs a day.\nShe makes 9 * 2 = $<<9*2=18>>18 every day at the farmer\u2019s market.\n#### 18|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)

## 可用数据集任务

### gsm8k_gen_ee684f
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|gsm8k_gen_ee684f|gsm8k数据集生成式任务|accuracy|4-shot|string|[gsm8k_gen_ee684f.py](gsm8k_gen_ee684f.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets gsm8k_gen_ee684f
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_ee684f import gsm8k_datasets
datasets = [
    *gsm8k_datasets,
]
```
**注:** 数据集任务的详细配置的含义请参见配置文件的注释
