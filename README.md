<div align="center">
  <br />
  <br />

  # **AISBench 评测工具**
  #### 面向人工智能领域的测试基准工具
  <!-- 用分隔线替代背景 -->
  ---

[![Release](https://img.shields.io/badge/dynamic/json?logo=gitee&label=release&logoColor=red&color=green&query=$.name&url=https://gitee.com/api/v5/repos/aisbench/benchmark/releases/latest)](https://gitee.com/aisbench/benchmark/releases)<br>
[![Gitee Forks](https://img.shields.io/badge/dynamic/json?logo=forgejo&label=forks&logoColor=blue&color=blue&query=$.forks_count&url=https://gitee.com/api/v5/repos/aisbench/benchmark?)](https://gitee.com/aisbench/benchmark/members)
[![Stars](https://img.shields.io/badge/dynamic/json?logo=ReverbNation&label=stars&logoColor=yellow&color=yellow&query=$.stargazers_count&url=https://gitee.com/api/v5/repos/aisbench/benchmark)](https://gitee.com/aisbench/benchmark/stargazers)
[![Gitee Issues](https://img.shields.io/badge/dynamic/json?logo=gitee&label=issues&logoColor=red&color=red&query=$.open_issues_count&url=https://gitee.com/api/v5/repos/aisbench/benchmark)](https://gitee.com/aisbench/benchmark/issues)<br>
[![License](https://img.shields.io/badge/license-Apache--2.0-red?logo=apache)](https://www.apache.org/licenses/LICENSE-2.0)
<br><br>
[🌐官方网站](https://www.aisbench.com) |
[📖工具文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/) |
[🔥最新进展](#-最新进展)|
[🤔报告问题](https://gitee.com/aisbench/benchmark/issues/new/choose)
<br><br>简体中文 | [English](README_en.md)
</div>

> ❗<span style="color: red;"><b>重要</b></span>
>
> **⭐️收藏项目**，你将能第一时间获取 AISBench评测工具 的最新动态～

## 🔥 最新进展
- **\[2025.9.08\]** 支持📚[模拟真实业务流量](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/rps_distribution.html)：通过控制请求发送速率波动，感知在模拟真实场景下服务化的性能测评结果！🔥🔥🔥

- **\[2025.8.28\]** 支持📚[多次独立重复推理精度场景](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id12)，计算pass@k/cons@k/avg@n等不同维度的精度指标！🔬🔬🔬

- **\[2025.8.19\]**
  - 新增Function Call专用模型配置 [vllm_api_function_call_chat](ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py)，支持 [BFCL 函数调用能力评估](ais_bench/benchmark/configs/datasets/BFCL/README.md) 🔥🔥🔥
  - 提供工具支持的[性能测试规格说明](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html#id25)，优化推理集群场景工具内存占用及性能计算。最大规格场景（250K条请求，输入/输出token 4K/4K）内存占用下降低60%，内存占用小于64GB；性能结果计算效率提升20倍。🚀🚀🚀
- **\[2025.7.15\]**
  - 支持[sharegpt](ais_bench/benchmark/configs/datasets/sharegpt/README.md)和[mtbench](ais_bench/benchmark/configs/datasets/mtbench/README.md)多轮对话数据集服务化性能测评和可视化，测评方式见📚[多轮对话测评指南](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/multiturn_benchmark.html)！🔥🔥🔥
  - 性能评测场景使用[自定义数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/advanced_tutorials/custom_dataset.html)，支持按请求粒度指定最大输出长度！🔥🔥🔥

- **\[2025.6.19\]** 支持📚[性能评测结果可视化](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/results_intro/performance_visualization.html)，辅助定位推理服务性能瓶颈！🔥🔥🔥
- **\[2025.6.12\]** 支持[textvqa](ais_bench/benchmark/configs/datasets/textvqa/README.md)、[videobench](ais_bench/benchmark/configs/datasets/videobench/README.md)和[vocalsound](ais_bench/benchmark/configs/datasets/vocalsound/README.md)等多模态数据集的精度和性能评测！🔥🔥🔥

- **\[2025.6.6\]** AISBench支持稳态性能评测，获取系统真实最佳性能，参考📚 [服务化稳定状态性能测试](doc/users_guide/stable_stage.md)进行快速上手! 🔥🔥🔥

- **\[2025.5.16\]** 支持3W+高并发服务化性能评测，📚 [性能指标](doc/users_guide/performance_metric.md)对齐🔗 [vllm benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)，参考📚 [服务化性能测评指南](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/performance_benchmark.html)了解详情！🔥🔥🔥

- **\[2025.4.30\]** 精度评测支持断点续测和失败用例重测，大幅提高精度评测鲁棒性，参考📚 [中断续测 & 失败用例重测](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id10)进行快速上手! 🔥🔥🔥

- **\[2025.4.15\]** 优化固定batch发送请求的方式为continuous batch模式发送请求，大幅提高精度评测效率! 🔥🔥🔥

- **\[2025.4.12\]** 支持合并MMLU、Ceval等所有多文件数据集为单个数据集任务进行精度评测，参考📚 [合并多文件数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id11)了解详情！ 🔥🔥🔥


## 🌏 简介
AISBench Benchmark 是基于 [OpenCompass](https://github.com/open-compass/opencompass) 构建的模型评测工具，兼容 OpenCompass 的配置体系、数据集结构与模型后端实现，并在此基础上扩展了对服务化模型的支持能力。

当前，AISBench 支持两大类推理任务的评测场景：

🔍 [精度测评](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/home.html#id2)：支持对服务化模型和本地模型在各类问答、推理基准数据集上的精度验证。

🚀 [性能测评](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/scenes_intro/home.html#id5)：支持对服务化模型的延迟与吞吐率评估，并可进行压测场景下的极限性能测试。

## 🛠️ 工具安装
✅ 环境要求

**Python 版本**：仅支持 Python **3.10** 或 **3.11**

不支持 Python 3.9 及以下，也不兼容 3.12 及以上版本

**推荐使用 Conda 管理环境**，以避免依赖冲突
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

📦 安装方式（源码安装）

AISBench 当前仅提供源码安装方式，请确保安装环境联网：
```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```
该命令会自动安装核心依赖。
执行`ais_bench -h`，如果打印出AISBench评测工具的所有命令行的帮助信息，说明安装成功

⚙️ 服务化框架支持（可选）

若需评估服务化模型（如 vLLM、Triton 等），需额外安装相关依赖：
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```
🔗 Berkeley Function Calling Leaderboard (BFCL) 测评支持

```shell
pip3 install -r requirements/bfcl_dependencies.txt --no-deps
```

**重要提示**：由于 `bfcl_eval` 会自动安装 `pathlib` 库，而 Python 3.5+ 环境已内置该库，为避免版本冲突，请务必使用 `--no-deps` 参数跳过额外依赖的自动安装。

如需进一步配置、使用 CLI 或 Python 脚本发起评测任务，请参考[快速入门指南](#快速入门)。

## ❌ 工具卸载
如需卸载 AISBench Benchmark，可执行以下命令：
```shell
pip3 uninstall ais_bench_benchmark
```

## 🚀 快速入门
### 命令含义
AISBench命令执行的单个或多个评测任务是由模型任务（单个或多个）、数据集任务（单个或多个）和结果呈现任务（单个）的组合定义的，AISBench的其他命令行则规定了评测任务的场景（精度评测场景、性能评测场景等）。以如下AISBench命令为例：
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```
此命令没有指定其他命令行，默认是一个精度评测场景的任务，其中：
- `--models`指定了模型任务，即`vllm_api_general_chat`模型任务。

- `--datasets`指定了数据集任务，即`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务。

- `--summarizer`指定了结果呈现任务，即`example`结果呈现任务(不指定`--summarizer`精度评测场景默认使用`example`任务)，一般使用默认，不需要在命令行中指定，后续命令不指定。

### 任务含义查询(可选)
所选模型任务`vllm_api_general_chat`、数据集任务`demo_gsm8k_gen_4_shot_cot_chat_prompt`和结果呈现任务`example`的具体信息(简介，使用约束等)可以分别从如下链接中查询含义：
- `--models`: 📚 [服务化推理后端](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/models.html#id2)

- `--datasets`: 📚 [开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html#id3) → 📚 [详细介绍](ais_bench/benchmark/configs/datasets/demo/README.md)

- `--summarizer`: 📚 [结果汇总任务](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/summarizer.html)

### 运行命令前置准备
- `--models`: 使用`vllm_api_general_chat`模型任务，需要准备支持`v1/chat/completions`子服务的推理服务，可以参考🔗 [VLLM启动OpenAI 兼容服务器](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server)启动推理服务
- `--datasets`: 使用`demo_gsm8k_gen_4_shot_cot_chat_prompt`数据集任务，需要准备gsm8k数据集，可以从🔗 [opencompass
提供的gsm8k数据集压缩包](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip)下载。将解压后的`gsm8k/`文件夹部署到AISBench评测工具根路径下的`ais_bench/datasets`文件夹下。

### 任务对应配置文件修改
每个模型任务、数据集任务和结果呈现任务都对应一个配置文件，运行命令前需要修改这些配置文件的内容。这些配置文件路径可以通过在原有AISBench命令基础上加上`--search`来查询，例如：
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
> ⚠️ **注意**： 执行带search命令会打印出任务对应的配置文件的绝对路径。

执行查询命令可以得到如下查询结果：
```shell
06/28 11:52:25 - AISBench - INFO - Searching configs...
╒══════════════╤═══════════════════════════════════════╤════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╕
│ Task Type    │ Task Name                             │ Config File Path                                                                                                               │
╞══════════════╪═══════════════════════════════════════╪════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╡
│ --models     │ vllm_api_general_chat                 │ /your_workspace/benchmark/ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py                                 │
├──────────────┼───────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ --datasets   │ demo_gsm8k_gen_4_shot_cot_chat_prompt │ /your_workspace/benchmark/ais_bench/benchmark/configs/datasets/demo/demo_gsm8k_gen_4_shot_cot_chat_prompt.py                   │
╘══════════════╧═══════════════════════════════════════╧════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╛

```

- 快速入门中数据集任务配置文件`demo_gsm8k_gen_4_shot_cot_chat_prompt.py`不需要做额外修改，数据集任务配置文件内容介绍可参考📚 [配置开源数据集](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/base_tutorials/all_params/datasets.html#id6)

模型配置文件`vllm_api_general_chat.py`中包含了模型运行相关的配置内容，是需要依据实际情况修改的。快速入门中需要修改的内容用注释标明。
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",
        model="DeepSeek-R1",        # 指定服务端已加载模型名称，依据实际VLLM推理服务拉取的模型名称配置（配置成空字符串会自动获取）
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",      # 指定推理服务的IP
        host_port = 8080,           # 指定推理服务的端口
        max_out_len = 512,
        batch_size=1,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]
```
### 执行命令
修改好配置文件后，执行命令启动服务化精度评测（⚠️ 第一次执行建议加上`--debug`，可以将具体日志打屏，如果有请求推理服务过程中的报错更方便处理）：
```bash
# 命令行加上--debug，
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
```
#### 查看任务执行细节
执行AISBench命令后，任务执行的细节会不断落盘在默认的输出路径，这个输出路径在运行中的打屏日志中有提示，例如：
```shell
06/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250628_151326
```

这段日志说明任务执行的细节落盘在执行命令的路径下的`outputs/default/20250628_151326`中。
命令执行结束后`outputs/default/20250628_151326`中的任务执行的细节如下所示：

```shell
20250628_151326/
├── configs # 模型任务、数据集任务和结构呈现任务对应的配置文件合成的一个配置
│   └── 20250628_151326_29317.py
├── logs # 执行过程中日志，命令中如果加--debug，不会有过程日志落盘（都直接打印出来了）
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # 基于predictions/文件夹下的推理结果的精度评测过程的日志
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # 推理过程日志
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 推理结果（推理服务返回的所有输出）
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # 精度评测计算的原始分数
└── summary
    ├── summary_20250628_151326.csv # 最终精度分数呈现（表格格式）
    ├── summary_20250628_151326.md # 最终精度分数呈现（markdown格式）
    └── summary_20250628_151326.txt # # 最终精度分数呈现（文本格式）
```
> ⚠️ **注意**： 不同评测场景落盘任务执行细节内容不同，具体请参考具体评测场景的指南。

#### 输出结果
因为只有8条数据，会很快跑出结果，结果显示的示例如下
```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```

更多教程请查看我们的👉[文档](https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/)


## 🔜 即将推出
- [ ] **\[2025.10\]** AISBench完成全面重构，支持在AISBench框架下🔌插件化集成前沿测试基准，以应对业界愈发复杂多样化的测试任务；并且显著提高易用性。
- [ ] **\[2025.11\]** 提供业界前沿的多模态测评能力。
- [ ] **\[2025.12\]** 提供业界主流Agent测评能力。
- [x] **\[2025.9\]** 支持模拟真实任务流量。
- [x] **\[2025.8\]** 将支持ShareGPT、BFCL等多轮对话数据集的性能评测。
- [x] **\[2025.8\]** 优化性能测评中评估eval阶段的计算效率，优化工具显存占用，补充工具使用规格说明。
- [x] **\[2025.7\]** 性能评测场景使用自定义数据集，将支持定义每条数据对应的最大输出长度限制。

## 🤝 致谢
- 本项目代码基于🔗 [OpenCompass](https://github.com/open-compass/opencompass)做拓展开发。
- 本项目部分数据集和提示词实现修改自[simple-evals](https://github.com/openai/simple-evals)。
- 本项目代码中打点的性能指标与[VLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks)对齐。
- 本项目的BFCL函数调用能力评估功能基于 [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) 实现。

<p align="right"><a href="#top">🔝Back to top</a></p>