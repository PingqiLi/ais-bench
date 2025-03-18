# AIME2024
## 数据集简介
AIME2024数据集包含了 2024 年美国数学邀请赛[（AIME）I 卷](https://artofproblemsolving.com/wiki/index.php/2024_AIME_I?srsltid=AfmBOoqP9aelPNCpuFLO2bLyoG9_elEBPgqcYyZAj8LtiywUeG5HUVfF)和 [(AIME)II 卷](https://artofproblemsolving.com/wiki/index.php/2024_AIME_II_Problems/Problem_15)中的 30 道题目。其原始来源是[AI-MO/aimo-validation-aime](https://hf-mirror.com/datasets/AI-MO/aimo-validation-aime)，该来源包含了一个更大的题目集，涵盖 2022 - 2024 年美国数学邀请赛的 90 道题目。

## 数据集原始获取链接
[https://huggingface.co/datasets/HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024)

## 数据集内容格式(处理后)
### 文件结构
```
aime
└── aime.jsonl
```
### 数据集内容样例格式
|index|origin_prompt|gold_answer|source|
| ----- | ---- | ---- | ---- |
|0|\nEvery morning, Aya does a $9$ kilometer walk, and then finishes at the coffee shop. One day, she walks at $s$ kilometers per hour, and the walk takes $4$ hours, including $t$ minutes at the coffee shop. Another morning, she walks at $s+2$ kilometers per hour, and the walk takes $2$ hours and $24$ minutes, including $t$ minutes at the coffee shop. This morning, if she walks at $s+\\frac12$ kilometers per hour, how many minutes will the walk take, including the $t$ minutes at the coffee shop?\n|204|aime2024|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/aime.zip)

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|aime2024_gen_0_shot_str|aime2024数据集生成式任务, 默认max out tokens长度取32768|accuracy(pass@1)|0-shot|string|[aime2024_gen_0_shot_str.py](aime2024_gen_0_shot_str.py)|