# mbpp
## 数据集简介
mbpp基准测试包含约1,000个众包Python编程题目，难度设计为入门级程序员可解决，涵盖编程基础、标准库功能等内容。每个题目包含任务描述、代码解决方案和3个自动化测试用例。如论文所述，我们已对部分数据进行了人工验证。

## 数据集原始获取链接
[https://huggingface.co/datasets/google-research-datasets/mbpp](https://huggingface.co/datasets/google-research-datasets/mbpp)

## 数据集内容格式(处理后)
### 文件结构
```
mbpp
├── mbpp.jsonl
└── sanitized-mbpp.jsonl
```
### 数据集内容样例格式
|text|code|task_id|test_setup_code|test_list|challenge_test_list|
| ----- | ---- | ---- | --- | --- | --- |
|Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].|R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]|1|""|["assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8", "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12", "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16"]|[]|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/mbpp.zip)
## 可用数据集任务
### mbpp_passk_gen_3_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mbpp_passk_gen_3_shot_chat_prompt|mbpp数据集生成式任务，支持测pass@k(默认pass@1)|pass@1|3-shot|对话格式|[mbpp_passk_gen_3_shot_chat_prompt.py](mbpp_passk_gen_3_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets mbpp_passk_gen_3_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.mbpp.mbpp_passk_gen_3_shot_chat_prompt import mbpp_datasets
datasets = [
    *mbpp_datasets,
]
```

### sanitized_mbpp_passk_gen_3_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|sanitized_mbpp_passk_gen_3_shot_chat_prompt|sanitized mbpp数据集生成式任务，支持测pass@k(默认pass@1)|pass@1|3-shot|对话格式|[sanitized_mbpp_passk_gen_3_shot_chat_prompt.py](sanitized_mbpp_passk_gen_3_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets sanitized_mbpp_passk_gen_3_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.mbpp.sanitized_mbpp_passk_gen_3_shot_chat_prompt import sanitized_mbpp_datasets
datasets = [
    *sanitized_mbpp_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释

