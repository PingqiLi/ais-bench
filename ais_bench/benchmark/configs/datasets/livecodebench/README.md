# LiveCodeBench
## 数据集简介
LiveCodeBench 是一个持续更新的"实时"基准测试平台，用于全面评估大语言模型（LLMs）的代码相关能力。该平台主要评估模型在代码生成、自我修复、测试输出预测和代码执行等多方面的表现。当前展示的是其代码生成场景，同时也被用于通过测试用例反馈来评估模型的自我修复能力。

该基准测试的问题采集自编程竞赛网站，特别注重保持题目质量、测试用例质量以及题目难度多样性。当前版本包含来自LeetCode、AtCoder和Codeforces的500余道题目。每个问题实例均包含题目描述、输入/输出示例及隐藏测试用例，且所有题目均标注难度等级和发布时间，便于测量模型在不同时间窗口下的表现。最终目标是针对每个问题实例生成正确且高效的解决方案。

初始版本的代码生成数据集因包含大量测试用例而导致数据集规模过大。当前（精简）版本在保持与原始数据集相似性能的前提下，对测试用例进行了筛选和抽样。未来LiveCodeBench将采用此精简版本进行代码生成评估。

## 数据集原始获取链接
[https://livecodebench.github.io/](https://livecodebench.github.io/)
## 数据集内容格式(处理后)
### 文件结构
```
code_generatation_lite
├── test5.jsonl
├── test4.jsonl
├── test3.jsonl
├── test2.jsonl
├── test1.jsonl
└── test.jsonl

```

### 处理后的数据集获取链接
[https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main][https://huggingface.co/datasets/livecodebench/code_generation_lite/tree/main]
## 可用数据集任务
### livecodebench_code_generate_lite_gen_0_shot_chat
#### 基本信息
|任务名称|简介|评估指标|few-shot|prompt格式|对应源码配置文件路径|
| --- | --- | --- | --- | --- | --- |
|livecodebench_code_generate_lite_gen_0_shot_chat|code_generatation_lite数据集生成式任务,|pass@1|0-shot|对话格式|[livecodebench_code_generate_lite_gen_0_shot_chat.py](livecodebench_code_generate_lite_gen_0_shot_chat.py)|

#### 命令行调用
```shell
ais_bench --models vllm_api_general_chat --datasets livecodebench_code_generate_lite_gen_0_shot_chat
```
#### 在自定义配置文件中导入
```python
from mmengine.config import read_base
with read_base():
    from ais_bench.benchmark.configs.datasets.livecodebench.livecodebench_code_generate_lite_gen_0_shot_chat import LCB_datasets
datasets = [
    *LCB_datasets,
]
```

**注:** 数据集任务的详细配置的含义请参见配置文件的注释