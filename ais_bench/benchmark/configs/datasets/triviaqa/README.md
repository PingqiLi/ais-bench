# TriviaQA
## 数据集简介
TriviaQA是一个阅读理解数据集，包含超过65万组"问题-答案-证据"三元组。该数据集包含9.5万道由 trivia 爱好者编写的问题-答案对，以及独立收集的佐证文档（平均每道问题6份），这些文档为问题解答提供了高质量的远程监督。。

## 数据集原始获取链接
[https://huggingface.co/datasets/mandarjoshi/trivia_qa](https://huggingface.co/datasets/mandarjoshi/trivia_qa)

## 数据集内容格式
### 文件结构
```
triviaqa
├── trivia-dev.qa.csv
├── triviaqa-train.jsonl
├── triviaqa-validation.jsonl
└── trivia-test.qa.csv
```
### 数据集内容样例格式
|question|answer|
| ----- | ---- |
|Which Lloyd Webber musical premiered in the US on 10th December 1993?|["Sunset Blvd", "West Sunset Boulevard", "Sunset Boulevard", "Sunset Bulevard", "Sunset Blvd."]|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/triviaqa.zip)

## 可用数据集任务
### triviaqa_gen_5_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|triviaqa_gen_5_shot_chat_prompt|TriviaQA数据集生成式任务|accuracy|5-shot|对话格式|[triviaqa_gen_5_shot_chat_prompt.py](triviaqa_gen_5_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets triviaqa_gen_5_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.triviaqa.triviaqa_gen_5_shot_chat_prompt import triviaqa_datasets
datasets = [
    *triviaqa_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释

