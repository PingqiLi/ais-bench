# BoolQ
## 数据集简介
BoolQ 是一个用于回答是非问题的问答数据集，包含 15942 个示例。这些问题是自然产生的 —— 它们是在无提示且不受限制的情况下生成的。每个示例都是由（问题、段落、答案）组成的三元组，页面标题作为可选的额外背景信息。
## 数据集原始获取链接
[https://huggingface.co/datasets/google/boolq](https://huggingface.co/datasets/google/boolq)
## 数据集内容格式(处理后)
### 文件结构
```
SuperGLUE/BoolQ/
├── test.jsonl
└── val.jsonl
```
### 数据集内容样例格式
|passage|question|idx|
| ---- | ---- | --- |
|20 euro note -- Until now there has been only one complete series of euro notes; however a new series, similar to the current one, is being released. The European Central Bank will, in due time, announce when banknotes from the first series lose legal tender status.|is the first series 20 euro note still legal tender|0|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/SuperGLUE.zip)

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|SuperGLUE_BoolQ_gen|SuperGLUE_BoolQ_gen数据集生成式任务|正确率(naive_average)|0-shot|string|[SuperGLUE_BoolQ_gen.py](SuperGLUE_BoolQ_gen_883d50_str.py)|
|SuperGLUE_BoolQ_cot_gen_1d56df_str.py|SuperGLUE_BoolQ_gen数据集生成式任务|正确率(naive_average)|0-shot|string|[SuperGLUE_BoolQ_cot_gen_1d56df_str.py](SuperGLUE_BoolQ_cot_gen_1d56df_str.py)|
|SuperGLUE_BoolQ_few_shot_gen_ba58ea_str.py|SuperGLUE_BoolQ_gen数据集生成式任务|正确率(naive_average)|5-shot|string|[SuperGLUE_BoolQ_few_shot_gen_ba58ea_str.py](SuperGLUE_BoolQ_few_shot_gen_ba58ea_str.py)|
