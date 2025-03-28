# mgsm
## 数据集简介
多语言小学数学能力测评基准（MGSM）是一个专注于小学数学题目的评估基准。

## 数据集原始获取链接
[https://huggingface.co/datasets/juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm)

## 数据集内容格式
### 文件结构
```
mgsm/
├── exemplars.py
├── mgsm_bn.tsv
├── mgsm_de.tsv
├── mgsm_en.tsv
├── mgsm_es.tsv
├── mgsm_fr.tsv
├── mgsm_ja.tsv
├── mgsm.py
├── mgsm_ru.tsv
├── mgsm_sw.tsv
├── mgsm_te.tsv
├── mgsm_th.tsv
├── mgsm_zh.tsv
└── README.md
```
### 数据集内容样例格式
|text|code|task_id|test_setup_code|test_list|challenge_test_list|
| ----- | ---- | ---- | --- | --- | --- |
|Write a function to find the minimum cost path to reach (m, n) from (0, 0) for the given cost matrix cost[][] and a position (m, n) in cost[][].|R = 3\r\nC = 3\r\ndef min_cost(cost, m, n): \r\n\ttc = [[0 for x in range(C)] for x in range(R)] \r\n\ttc[0][0] = cost[0][0] \r\n\tfor i in range(1, m+1): \r\n\t\ttc[i][0] = tc[i-1][0] + cost[i][0] \r\n\tfor j in range(1, n+1): \r\n\t\ttc[0][j] = tc[0][j-1] + cost[0][j] \r\n\tfor i in range(1, m+1): \r\n\t\tfor j in range(1, n+1): \r\n\t\t\ttc[i][j] = min(tc[i-1][j-1], tc[i-1][j], tc[i][j-1]) + cost[i][j] \r\n\treturn tc[m][n]|1|""|["assert min_cost([[1, 2, 3], [4, 8, 2], [1, 5, 3]], 2, 2) == 8", "assert min_cost([[2, 3, 4], [5, 9, 3], [2, 6, 4]], 2, 2) == 12", "assert min_cost([[3, 4, 5], [6, 10, 4], [3, 7, 5]], 2, 2) == 16"]|[]|

### 处理后的数据集获取链接
[https://huggingface.co/datasets/juletxara/mgsm](https://huggingface.co/datasets/juletxara/mgsm)
```shell
git clone https://huggingface.co/datasets/juletxara/mgsm
```
## 可用数据集任务
### mgsm_gen_0_shot_cot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mgsm_gen_0_shot_cot_chat_prompt|mgsm数据集生成式任务，prompt带逻辑链|accuracy|0-shot|对话格式|[mgsm_gen_0_shot_cot_chat_prompt.py](mgsm_gen_0_shot_cot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets mgsm_gen_0_shot_cot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.mgsm.mgsm_gen_0_shot_cot_chat_prompt import mgsm_datasets
datasets = [
    *mgsm_datasets,
]
```

### mgsm_gen_8_shot_cot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|mgsm_gen_8_shot_cot_chat_prompt|mgsm数据集生成式任务，prompt带逻辑链|accuracy|8-shot|对话格式|[mgsm_gen_8_shot_cot_chat_prompt.py](mgsm_gen_8_shot_cot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets mgsm_gen_8_shot_cot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.mgsm.mgsm_gen_8_shot_cot_chat_prompt import mgsm_datasets
datasets = [
    *mgsm_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释

