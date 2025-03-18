# GPQA
## 数据集简介
GPQA 是一个包含选择题的问答数据集，其中的高难度问题由生物学、物理学和化学领域的专家编写并验证。当这些专家尝试回答自己专业领域之外的问题时（例如，一位物理学家回答化学问题），即便他们能无限制地使用谷歌搜索，且花费超过 30 分钟作答，答题准确率也仅有 34%。

## 数据集原始获取链接
[https://github.com/idavidrein/gpqa](https://github.com/idavidrein/gpqa)

## 数据集内容格式(处理后)
### 文件结构
```
gpqa
├── gpqa_diamond.csv
├── gpqa_experts.csv
├── gpqa_extended.csv
├── gpqa_main.csv
└── license.txt
```
### 数据集内容样例格式
| Pre-Revision Question                                                                                                                                                                                                                                               | Pre-Revision Correct Answer | Pre-Revision Incorrect Answer 1 | Pre-Revision Incorrect Answer 2 | Pre-Revision Incorrect Answer 3 | Pre-Revision Explanation                                                                                                                                                                                                                | Self-reported question-writing time (minutes) | Question                                                                                                                                                                                                                                                                | Correct Answer | Incorrect Answer 1 | Incorrect Answer 2 | Incorrect Answer 3 | Explanation                                                                                                                                                                                                                              |
|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------:|:-------------------------------:|:-------------------------------:|:-------------------------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------------------------------------:|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------:|:------------------:|:------------------:|:------------------:|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they be clearly resolved? | 10^-4 ev                    | 10^-8 ev                        | 10^-9 ev                        | 10^-11 ev                       | "According to uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the width of the energy level . With Delta t=10^-9 s==> Delta E1= 3.3 *10^-7 ev. And Delta t=10^-11 s gives Delta E2=3.3*10^-8 ev. | 30                                            | "Two quantum states with energies E1 and E2 have a lifetime of 10^-9 sec and 10^-8 sec, respectively. We want to clearly distinguish these two energy levels. Which one of the following options could be their energy difference so that they can be clearly resolved? | 10^-4 eV       | 10^-11 eV          | "10^-8 eV          | 10^-9 eV           | "According to the uncertainty principle, Delta E* Delta t=hbar/2. Delta t is the lifetime and Delta E is the width of the energy level. With Delta t=10^-9 s==> Delta E1= 3.3 10^-7 ev. And Delta t=10^-11 s gives Delta E2=3.310^-8 eV. |

### 处理后的数据集获取链接
[http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gpqa.zip)

## 可用数据集任务
### gpqa_gen_0_shot_str
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|gpqa_gen_0_shot_str|gpqa数据集生成式任务, 默认max out tokens长度取32768|accuracy(pass@1)|0-shot|string|[gpqa_gen_0_shot_str.py](gpqa_gen_0_shot_str.py)|
#### 命令行调用
```shell
ais_bench --models vllm_api_general --datasets gpqa_gen_0_shot_str
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_str import gpqa_datasets
datasets = [
    *gpqa_datasets,
]
```
**注:** 数据集任务的详细配置的含义请参见配置文件的注释

