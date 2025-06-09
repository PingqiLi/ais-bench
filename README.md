# AISBench Benchmark评测工具
## 简介
AISBench Benchmark 是基于 OpenCompass 构建的模型评测工具，兼容 OpenCompass 的配置体系、数据集结构与模型后端实现，并在此基础上扩展了对服务化模型的支持能力。

当前，AISBench 支持两大类推理任务的评测场景：

✅ [精度评测](#精度测评)：支持对服务化模型和本地模型在各类问答、推理基准数据集上的精度验证。

🚀 [性能评测](#性能测评)：支持对服务化模型的延迟与吞吐率评估，并可进行压测场景下的极限性能测试。

## 工具安装
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
pip3 install -e ./
```
该命令会自动安装核心依赖。

⚙️ 服务化框架支持（可选）

若需评估服务化模型（如 vLLM、Triton 等），需额外安装相关依赖：
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```
如需进一步配置、使用 CLI 或 Python 脚本发起评测任务，请参考[快速入门指南](#快速入门)。
## 工具卸载
如需卸载 AISBench Benchmark，可执行以下命令：
```shell
pip3 uninstall ais_bench_benchmark
```

## 快速入门
在 AISBench 中，每个评测任务由模型后端和数据集共同定义。支持两种配置方式：

[命令行界面CLI指定模型和数据集](#命令行界面CLI指定模型和数据集)：适用于简单场景，启动便捷

[Python配置文件指定模型和数据集](#Python配置文件指定模型和数据集)：适用于复杂或批量评测任务，配置灵活可复用

> ⚠️ 注意： 所有评测任务依赖预配置的数据集。请参考 [数据集准备指南](./doc/users_guide/datasets.md#数据集准备指南) 进行数据集配置。

### 命令行界面CLI指定模型和数据集
#### 命令说明
在任意路径下执行CLI命令：
  ```bash
    # 命令行界面 (CLI)
    ais_bench --models {模型配置名称} --datasets {数据集配置名称} [OPTIONS]
    #示例：ais_bench --models vllm_api_general --datasets gsm8k_gen
```
基本参数说明：
- `--models`: 模型配置名称，支持的模型配置可参考：[模型配置说明](./doc/users_guide/models.md#模型配置说明)
- `--datasets`: 数据集配置名称，请先按照[数据集准备指南](./doc/users_guide/datasets.md#数据集准备指南)完成所需数据集的配置

`[OPTIONS]`为ais_bench的可选参数，若想进行更加自定义的评测，例如中断续推、结果可视化或日志调试可以查看：[用户配置参数](doc/users_guide/cli_args.md#用户配置参数)
#### 示例：vLLM 服务化后端测评
测评vLLM 服务化后端为例，参考[服务化推理后端](./doc/users_guide/models.md#服务化推理后端)修改其 `v1/chat/completions` 子服务的模型配置文件`vllm_api_general_chat.py`
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",
        model="DeepSeek-R1",        # 指定服务端已加载模型名称
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
修改好配置文件后，执行如下命令启动精度评测：
```bash
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt
```

#### 输出结果
```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```

### Python配置文件指定模型和数据集
#### 使用说明
```bash
ais_bench ais_bench/configs/{模型类型}_examples/{任务配置文件名}
# 示例：
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
  ```
#### 测试样例
以下示例展示如何同时评测两个服务接口（[`v1/chat/completions`](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) 与 [`v1/completions`](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)）在 [GSM8K](ais_bench/benchmark/configs/datasets/gsm8k/README.md) 与 [MATH数据集](ais_bench/benchmark/configs/datasets/math/README.md)上的表现。参考示例：[demo_infer_vllm_api.py](ais_bench/configs/api_examples/demo_infer_vllm_api.py)：

```python
from mmengine.config import read_base
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask
from ais_bench.benchmark.models import VLLMCustomAPIChat

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.math.math500_gen_0_shot_cot_chat_prompt import math_datasets as math500_gen_0_shot_cot_chat
    from ais_bench.benchmark.configs.models.vllm_api.vllm_api_general import models as vllm_api_general

# 只取部分样本进行 demo 测试
gsm8k_0_shot_cot_str[0]['abbr'] = 'demo_' + gsm8k_0_shot_cot_str[0]['abbr']
gsm8k_0_shot_cot_str[0]['reader_cfg']['test_range'] = '[0:8]'

math500_gen_0_shot_cot_chat[0]['abbr'] = 'demo_' + math500_gen_0_shot_cot_chat[0]['abbr']
math500_gen_0_shot_cot_chat[0]['reader_cfg']['test_range'] = '[0:8]'

datasets = gsm8k_0_shot_cot_str + math500_gen_0_shot_cot_chat # 指定数据集列表，可通过累加添加不同的数据集配置
models = [      # 指定模型配置列表
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='demo-vllm-api-general-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",  # 指定推理服务的IP
        host_port = 8080,       # 指定推理服务的端口
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

models += vllm_api_general # 可组合不同的API模型

work_dir = 'outputs/demo_api-vllm-general-chat/'
```
修改好配置文件后，执行如下命令启动精度评测：
```
ais_bench ais_bench/configs/api_examples/demo_infer_vllm_api_general_chat.py
```
#### 输出结果
```bash
dataset                 version  metric   mode  demo-vllm-api-general-chat demo-vllm-api-general
----------------------- -------- -------- ----- -------------------------- ---------------------
demo_gsm8k              401e4c   accuracy gen                     62.50                62.50
demo_math_prm800k_500   c4b6f0   accuracy gen                     50.00                62.50
```

#### 预设配置文件
你可以在 `ais_bench/configs/` 文件夹下找到更多的Python脚本示例，详情参考:[Python自定义配置文件样例列表](./doc/users_guide/python_examples.md)。
### 推理过程查看
评测过程中，日志默认保存在：
```swift
{work_dir}/{time_label}/logs/infer/{abbr_name}/{dataset}.out
```
AISBench Benchmark的执行日志默认直接保存在日志文件中，启动推理后可以在{work_dir}/{time_label}/logs/infer/{abbr_name}/gsm8k.out 中查看推理结果，例如执行
查看方式示例：
```shell
# CLI 示例
tail -f outputs/default/20250126_165049/logs/infer/vllm-api-general/gsm8k.out

# Python 脚本示例
tail -f outputs/api_vllm_general/20250126_165049/logs/infer/vllm-api-general/gsm8k.out
```
其中`{work_dir}/{time_label}/`会在工具的打屏中进行提示，示例：
```bash
06/08 15:31:14 - AISBench - INFO - Current exp folder: outputs/demo-api-vllm-general-chat/20250608_160222
```
你也可以在命令中添加 `--debug` 参数将日志直接输出到控制台。更多细节请见:[用户配置参数](doc/users_guide/cli_args.md#用户配置参数)

## 精度测评
### 服务化精度测评
- 功能描述：评估部署为服务形式的模型在特定数据集上的预测准确率

- 要求：模型已部署，需测试其实际服务能力

- 支持：

    - 数据集：[开源数据集](doc/users_guide/datasets.md#开源数据集) 与 [自定义数据集](doc/users_guide/datasets.md#自定义数据集)

    - 模型后端：[服务化推理后端](doc/users_guide/models.md#服务化推理后端)

📚 详见文档：[服务化精度测评指南](doc/users_guide/accuracy_benchmark.md#服务化精度测评)

### 纯模型精度测评
- 功能描述：评估本地加载模型（非服务化）在不同数据集上的准确性

- 要求：离线模型权重和部署环境

- 支持：

    - 数据集：[开源数据集](doc/users_guide/datasets.md#开源数据集) 与 [自定义数据集](doc/users_guide/datasets.md#自定义数据集)

    - 模型后端：[本地模型后端](doc/users_guide/models.md#本地模型后端)

📚 详见文档：[纯模型精度测评指南](doc/users_guide/accuracy_benchmark.md#纯模型精度测评)

## 性能测评
### 服务化性能测评
- 功能描述：在真实部署环境中评估服务模型的运行效率（吞吐、延迟）

- 要求：模型接口需支持流式请求

- 支持：

    - 数据集：[支持数据集类型](doc/users_guide/datasets.md#支持数据集类型)中的所有数据类型

    - 模型后端：[服务化推理后端](doc/users_guide/models.md#服务化推理后端)

📚 详见文档：[服务化性能测评指南](doc/users_guide/performance_benchmark.md#服务化性能测评指南)。

### 服务化性能压测
- 功能描述：在最大并发场景下评估服务模型的运行效率（吞吐、延迟）

- 要求：

    - 支持流式输出接口

    - 明确设置最大并发参数

- 支持：

    - 数据集：[支持数据集类型](doc/users_guide/datasets.md#支持数据集类型)中的所有数据类型

    - 模型后端：[服务化推理后端](doc/users_guide/models.md#服务化推理后端)

📚 详见文档：[服务化性能压力测试指南](doc/users_guide/pressure_performance_benchmark.md)
