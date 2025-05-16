# BBH
## 数据集简介
BIG-Bench（Srivastava等人，2022年）是一个多样化的评估测试集，其重点关注当前语言模型被认为尚无法完成的任务。尽管语言模型已在该基准测试中取得显著进展——BIG-Bench论文中的最佳模型通过少量示例提示（few-shot prompting），在65%的任务上超越了人类评估者的平均成绩。但究竟在哪些任务上语言模型仍落后于人类平均水平？这些任务是否真的超出了当前语言模型的解决能力？

## 数据集原始获取链接
[https://huggingface.co/datasets/lukaemon/bbh](https://huggingface.co/datasets/lukaemon/bbh)

## 数据集内容格式(处理后)
### 文件结构
```
BBH
├── data
│   ├── boolean_expressions.json
│   ├── causal_judgement.json
│   ├── date_understanding.json
│   ├── disambiguation_qa.json
│   ├── dyck_languages.json
│   ├── formal_fallacies.json
│   ├── geometric_shapes.json
│   ├── hyperbaton.json
│   ├── logical_deduction_five_objects.json
│   ├── logical_deduction_seven_objects.json
│   ├── logical_deduction_three_objects.json
│   ├── movie_recommendation.json
│   ├── multistep_arithmetic_two.json
│   ├── navigate.json
│   ├── object_counting.json
│   ├── penguins_in_a_table.json
│   ├── README.md
│   ├── reasoning_about_colored_objects.json
│   ├── ruin_names.json
│   ├── salient_translation_error_detection.json
│   ├── snarks.json
│   ├── sports_understanding.json
│   ├── temporal_sequences.json
│   ├── tracking_shuffled_objects_five_objects.json
│   ├── tracking_shuffled_objects_seven_objects.json
│   ├── tracking_shuffled_objects_three_objects.json
│   ├── web_of_lies.json
│   └── word_sorting.json
└── lib_prompt
    ├── boolean_expressions.txt
    ├── causal_judgement.txt
    ├── date_understanding.txt
    ├── disambiguation_qa.txt
    ├── dyck_languages.txt
    ├── formal_fallacies.txt
    ├── geometric_shapes.txt
    ├── hyperbaton.txt
    ├── logical_deduction_five_objects.txt
    ├── logical_deduction_seven_objects.txt
    ├── logical_deduction_three_objects.txt
    ├── movie_recommendation.txt
    ├── multistep_arithmetic_two.txt
    ├── navigate.txt
    ├── object_counting.txt
    ├── penguins_in_a_table.txt
    ├── reasoning_about_colored_objects.txt
    ├── ruin_names.txt
    ├── salient_translation_error_detection.txt
    ├── snarks.txt
    ├── sports_understanding.txt
    ├── temporal_sequences.txt
    ├── tracking_shuffled_objects_five_objects.txt
    ├── tracking_shuffled_objects_seven_objects.txt
    ├── tracking_shuffled_objects_three_objects.txt
    ├── web_of_lies.txt
    └── word_sorting.txt
```

### 数据集内容样例格式
|input|target|
| ----- | ------- |
|Today is Christmas Eve of 1937. What is the date tomorrow in MM/DD/YYYY?\nOptions:\n(A) 12/11/1937\n(B) 12/25/1937\n(C) 01/04/1938\n(D) 12/04/1937\n(E) 12/25/2006\n(F) 07/25/1937|(B)|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/BBH.zip)

## 可用数据集任务
### bbh_gen_3_shot_cot_chat
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|bbh_gen_3_shot_cot_chat|BBH数据集生成式任务|score(accuracy)|3-shot|对话格式|[bbh_gen_3_shot_cot_chat.py](bbh_gen_3_shot_cot_chat.py)|

#### 命令行调用
```shell
# 每个数据集文件作为单一task来评测
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets bbh_gen_3_shot_cot_chat

# 合并多个数据集文件统一评测
## 对话格式后端
ais_bench --models vllm_api_general_chat --datasets bbh_gen_3_shot_cot_chat --merge-ds
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.bbh.bbh_gen_3_shot_cot_chat import bbh_datasets
datasets = [
    *bbh_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见Python源码配置文件的注释