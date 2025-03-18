# MATH
## 数据集简介
MATH 是一个包含 12500 道具有挑战性的竞赛数学题的新数据集。MATH 数据集中的每一道题都配有完整的分步解答，可用于训练模型生成答案推导过程和解释内容。

## 数据集原始获取链接
[https://github.com/hendrycks/math/](https://github.com/hendrycks/math/)

## 数据集内容格式(处理后)
### 文件结构
```
math
├── convert_jsonl2json.py
├── math.json
├── test.jsonl
├── test_prm800k_500.json # MATH500
├── test_prm800k_500.jsonl # MATH500
└── train.jsonl

```
### 数据集内容样例格式
|problem|solution|subject|level|unique_id|
| ---- | ---- | ----- | ----- | ----- |
|"What is $10.0000198\\cdot 5.9999985401\\cdot 6.9999852$ to the nearest whole number?|Notice that $10.00001988$ is very close to $10$, $5.9999985401$ is very close to $6$ and $6.9999852$ is very close to $7$. Because the given numbers are all so close to integers, we're unlikely to go wrong by rounding before multiplying. We get $$10\\cdot6\\cdot7=\\boxed{420}.$$If we multiplied the given numbers with a calculator we would get $$6.9999852\\cdot5.9999985401\\cdot10.00001988=419.999844...$$which would still round to $420$.|Prealgebra|Level 3|test/prealgebra/126.json|

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/math.zip)

## 可用数据集任务
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|math_prm800k_500_0shot_cot_gen|MATH500数据集生成式任务, 默认max out tokens长度取32768，prompt带逻辑链|accuracy(pass@1)|0-shot|string|[math_prm800k_500_0shot_cot_gen.py](math_prm800k_500_0shot_cot_gen.py)|
|math_prm800k_500_5shot_cot_gen|MATH500数据集生成式任务, 默认max out tokens长度取32768，prompt带逻辑链|accuracy(pass@1)|5-shot|string|[math_prm800k_500_5shot_cot_gen.py](math_prm800k_500_5shot_cot_gen.py)|