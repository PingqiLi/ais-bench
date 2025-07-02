# DEMO
## 数据集简介
此数据集用于文档快速入门使用，截取GSM8K数据集的前8条进行测试。
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

### demo_gsm8k_gen_4_shot_cot_chat_prompt
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|demo_gsm8k_gen_4_shot_cot_chat_prompt|gsm8k数据集生成式任务(只取8条数据)，带逻辑链|accuracy|4-shot|字符串格式|[demo_gsm8k_gen_4_shot_cot_chat_prompt.py](demo_gsm8k_gen_0_shot_cot_str_perf.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_0_shot_cot_str_perf --mode all
```

### demo_gsm8k_gen_0_shot_cot_str_perf
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|demo_gsm8k_gen_0_shot_cot_str_perf|gsm8k数据集生成式任务(只取8条数据)，带逻辑链|性能评测|0-shot|字符串格式|[demo_gsm8k_gen_0_shot_cot_str_perf.py](demo_gsm8k_gen_0_shot_cot_str_perf.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets demo_gsm8k_gen_0_shot_cot_str_perf --mode perf
```

