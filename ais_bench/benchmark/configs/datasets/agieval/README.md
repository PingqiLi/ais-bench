# AGIEval
## 数据集简介
AGIEval—是一个专为评估基础模型而设计的新型基准测试，其特别关注人类中心化的标准化考试场景，包括大学入学考试、法学院入学测试、数学竞赛以及律师资格考试等。

## 数据集原始获取链接
[https://github.com/ruixiangcui/AGIEval](https://github.com/ruixiangcui/AGIEval)

## 数据集内容格式(处理后)
### 文件结构
```
agieval/
├── aqua-rat.jsonl
├── gaokao-biology.jsonl
├── gaokao-chemistry.jsonl
├── gaokao-chinese.jsonl
├── gaokao-english.jsonl
├── gaokao-geography.jsonl
├── gaokao-history.jsonl
├── gaokao-mathcloze.jsonl
├── gaokao-mathqa.jsonl
├── gaokao-physics.jsonl
├── jec-qa-ca.jsonl
├── jec-qa-kd.jsonl
├── LICENSE
├── logiqa-en.jsonl
├── logiqa-zh.jsonl
├── lsat-ar.jsonl
├── lsat-lr.jsonl
├── lsat-rc.jsonl
├── math.jsonl
├── sat-en.jsonl
├── sat-en-without-passage.jsonl
└── sat-math.jsonl
```
### 数据集内容样例格式
|passage|question|options|label|answer|other|
| ----- | ---- | ---- | --- | --- | --- |
|null|Find out which of the following values is the multiple of X, if it is divisible by 9 and 12?|["(A)36", "(B)15", "(C)17", "(D)5", "(E)7"]|A|null|{"solution": "9=3*3\n12=3*4\nThe number should definitely have these factors 3*3*4\n36 is the number that has these factors\nSo, 36 is the multiple of X\nAnswer is A"}|

### 处理后的数据集获取链接
[https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
将其中`data/AGIEval/data/v1`下的文件复制到`agieval/`中

## 可用数据集任务
### agieval_gen_0_shot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|agieval_gen_0_shot_chat_prompt|AGIEval数据集生成式任务，共包含21个子任务|accuracy|0-shot|对话格式|[agieval_gen_0_shot_chat_prompt.py](agieval_gen_0_shot_chat_prompt.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets agieval_gen_0_shot_chat_prompt
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.agieval.agieval_gen_0_shot_chat_prompt import agieval_datasets
datasets = [
    *agieval_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释