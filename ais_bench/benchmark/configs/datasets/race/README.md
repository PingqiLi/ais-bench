# race
## 数据集简介
RACE（Reading Comprehension from Examinations）数据集是一个大规模的机器阅读理解数据集。该数据集由中国12-18岁学生的英语考试题目构成，包含27933篇文章和97867个问题。RACE数据集分为两个子集：RACE-M和RACE-H，分别对应初中和高中的题目难度。RACE-M包含28293个问题，适合初中生水平；RACE-H包含69574个问题，适合高中生水平。每个问题都有四个备选答案，其中一个是正确答案。
## 数据集原始获取链接
[https://huggingface.co/datasets/ehovy/race](https://huggingface.co/datasets/ehovy/race)
## 数据集内容格式(处理后)
### 文件结构
```
race/
├── test/
├───── high.jsonl
├───── middle.jsonl
├── validation/
├───── high.jsonl
├───── middle.jsonl
```
### 数据集内容样例格式
|example_id|article|answer|question|options|
| ---- | ---- | ---- | ---- | ---- |
|middle2177.txt|It is well-known that the \"prom\", a formal dance held at the end of high school or college, is an important date in every student's life. What is less well-known is that the word \"prom\" comes from the verb \"to promenade\", which means to walk around, beautifully dressed, in order to attract attention. The idea is that you should see and be seen by others.\nThe prom is not just an American tradition, though most people believe that it started in America. In Canada the event is called a \"formal\". In Britain and Australia the old fashioned word \"dance\" is more and more frequently being referred to as a \"prom\". Most countries have some form of celebration when students finish high school: after all, it means the end of life as a child, and the beginning of life as an adult.\nThe prom is expensive to organize and the tickets can cost students a lot of money. The tradition is that students themselves have to raise the money to pay for it. Selling the students newspapers is one way to raise money; so is taking a part-time job at the weekend.\nAlthough the prom should be the experience of a lifetime, it also worries many students. There is the problem of what to wear, who to take as your partner, who will be voted \"prom queen\", etc. And it is not only students who feel worried. What many parents find difficult is the realization that their children's school days are almost over. However, for most people at the prom, once the music starts playing, everyone relaxes and stops worrying.|"B"|"In which country is the prom called a \"formal\"?"|["America.", "Canada.", "Britain.", "Australia."]|

### 处理后的数据集获取链接
[https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip](https://github.com/open-compass/opencompass/releases/download/0.2.2.rc1/OpenCompassData-core-20240207.zip)
其中`data/race/`就是处理好的数据集

## 可用数据集任务

### race_middle_gen_5_shot_chat
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|race_middle_gen_5_shot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_middle_gen_5_shot_chat.py](race_middle_gen_5_shot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets race_middle_gen_5_shot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.race.race_middle_gen_5_shot_chat import race_datasets
datasets = [
    *race_datasets,
]
```

### race_middle_gen_5_shot_cot_chat
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|race_middle_gen_5_shot_cot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_middle_gen_5_shot_cot_chat.py](race_middle_gen_5_shot_cot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets race_middle_gen_5_shot_cot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.race.race_middle_gen_5_shot_cot_chat import race_datasets
datasets = [
    *race_datasets,
]
```

### race_high_gen_5_shot_chat
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|race_high_gen_5_shot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_high_gen_5_shot_chat.py](race_high_gen_5_shot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets race_high_gen_5_shot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.race.race_high_gen_5_shot_chat import race_datasets
datasets = [
    *race_datasets,
]
```

### race_high_gen_5_shot_cot_chat
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|race_high_gen_5_shot_cot_chat|race数据集生成式任务|accuracy|5-shot|对话格式|[race_high_gen_5_shot_cot_chat.py](race_high_gen_5_shot_cot_chat.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets race_high_gen_5_shot_cot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.race.race_high_gen_5_shot_cot_chat import race_datasets
datasets = [
    *race_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释