# 服务化精度测评
在服务化部署环境，通过标准化请求对比模型输出与标准答案，评估实际服务场景下的准确率。支持多种数据集与后端配置，便于对比不同服务化方案的模型精度。
## 测试准备
在执行服务化推理前，需要满足以下条件：

- 可访问的服务化模型服务：确保服务进程可在当前环境下直接访问。
- 数据集准备：支持使用[开源数据集](datasets.md#开源数据集)或[自定义数据集](datasets.md#自定义数据集)。建议将开源数据集手动放置在默认目录 `ais_bench/datasets/`下，程序将在任务执行时自动加载数据集文件。
- 服务化模型后端配置：请参考[服务化推理后端](models.md#服务化推理后端)进行后端配置，详细参数说明见[服务化推理后端配置参数说明](models.md#服务化推理后端配置参数说明)。

## 主要功能
服务化精度测评场景下主要包含以下功能：
- [多任务测评](#多任务测评)：支持同时配置多个模型或多个数据集任务，通过单次命令进行批量测评，适用于大规模模型横向对比或多数据集精度对比分析。


- [多任务并行测评](#多任务并行测评)：支持创建多个进程同时执行不同的测评任务，适用于串行任务执行效率不足的场景。


- [中断续测 & 失败用例重测](#中断续测--失败用例重测)：支持从推理任务中断位置继续推理剩余请求，或对失败用例重新推理。适用于Benchmark过程意外中断或服务器异常导致的推理任务失败场景。


- [合并子数据集推理](#合并子数据集推理)：支持包含多个子类别数据集合并推理。适用于验证模型在多样本来源下的泛化能力测试场景。

### 多任务测评
用户可通过`--models`和`--datasets`参数指定多个配置任务，子任务数为`--models`配置任务数和`--datasets`配置任务数的乘积，即一个模型配置和一个数据集配置组成一个子任务，示例：
```bash
ais_bench --models vllm_api_general vllm_api_stream_chat --datasets gsm8k_gen math500_gen_0_shot_cot_chat_prompt
```
示例中使用`v1/completions`和`v1/chat/completions`对[`GSM8K`](../../ais_bench/benchmark/configs/datasets/gsm8k/README.md)和[`MATH`](../../ais_bench/benchmark/configs/datasets/math/README.md)数据集进行测试，总共4个任务，任务结束后会在[`--work-dir`](cli_args.md#公共参数)生成如下文件内容：
```bash
outputs/default/
└── 20230220_183030     # 任务创建时间对应的输出目录
    ├── configs         # 存储已转储的配置文件。如果在同一实验目录下重复执行不同配置的任务，可能会存在多个配置文件
    │    └── 20230220_183030_2607108.py
    ├── logs            # 包含推理与精度评估阶段的日志
    │   ├── eval        # 精度计算阶段日志
    │   │   ├── vllm-api-general
    │   │   │   ├── gsm8k.out
    │   │   │   └── math_prm800k_500.out    
    │   │   └── vllm_api_stream_chat
    │   │       ├── gsm8k.out
    │   │       └── math_prm800k_500.out  
    │   └── infer       # 推理阶段日志
    │       ├── vllm-api-general
    │       │   ├── gsm8k.out
    │       │   └── math_prm800k_500.out    
    │       └── vllm_api_stream_chat
    │           ├── gsm8k.out
    │           └── math_prm800k_500.out  
    ├── predictions     # 推理结果文件，记录每条请求的输入、模型输出及参考答案（用于精度计算）
    │   ├── vllm-api-general
    │   │   ├── gsm8k.json
    │   │   └── math_prm800k_500.json    
    │   └── vllm_api_stream_chat
    │       ├── gsm8k.json
    │       └── math_prm800k_500.json  
    ├── results         # 基于 predictions 生成的精度评估结果
    │   ├── vllm-api-general
    │   │   ├── gsm8k.json
    │   │   └── math_prm800k_500.json    
    │   └── vllm_api_stream_chat
    │       ├── gsm8k.json
    │       └── math_prm800k_500.json 
    └── summary         # 精度结果的汇总视图，包含 CSV、Markdown 和 TXT 格式
        ├── summary_20230220_183030.csv
        ├── summary_20230220_183030.md
        └── summary_20230220_183030.txt

```
### 多任务并行测评
默认情况下，多个子任务采用串行执行，单个任务内默认开启Continous Batch，会根据用户配置的最大并发拉起多个进程发送和处理请求，允许配置较大的并发。在单个任务并发较小时，可以通过设置[`--max-num-workers`](cli_args.md#精度测评参数)参数实现多任务并行，示例如下：


> ⚠️ 注意：启用 --max-num-workers 时必须关闭 Continous Batch（通过 --disable-cb 参数），此时每个任务建议最大并发不超过 500。
```bash
ais_bench --models vllm_api_general vllm_api_stream_chat --datasets gsm8k_gen math500_gen_0_shot_cot_chat_prompt --disable-cb --max-num-workers 4
```
示例中指定任务最大并发数为4，四个子任务将会同时执行，生成结果与[多任务测评](#多任务测评)示例一致。

### 中断续测 & 失败用例重测
在测评过程中发生中断或部分任务失败时，可通过`--reuse`开启断点管理功能实现任务续测，亦支持仅对失败用例进行自动重测，无需重复运行全部任务。示例如下：


1、当用户使用如下命令首次推理时，由于任务异常退出导致的任务中断或由于服务端异常导致部分请求失败
```bash
ais_bench --models vllm_api_general --datasets gsm8k_gen
```
此时部分推理结果会被保存下来，在[`--work-dir`](cli_args.md#公共参数)生成如下文件内容：

```bash
outputs/default/
└── 20230220_183030     # 测试任务创建的时间戳目录
    ├── configs         # 存储已转储的配置文件。若在该目录下重复执行不同配置的实验，可能会生成多个配置文件
    │    └── 20230220_183030_2607108.py
    ├── logs            # 包含推理阶段与精度计算阶段的日志文件
    │   └── infer       # 推理阶段日志
    └── predictions     # 推理结果目录，记录每条请求的输入、模型输出及答案（用于精度评估）
        └── vllm-api-general
            └── tmp_gsm8k   # 已完成请求的推理输出
                └── tmp_0_2766386_1749107195.json   # 缓存文件，命名格式为：tmp_{任务进程ID}_{进程编号}_{时间戳}.json
```
2、通过`--reuse`参数指定任务时间戳目录续推：
```bash
ais_bench --models vllm_api_general --datasets gsm8k_gen --reuse 20230220_183030
```
日志中会打印如下内容，提示续推任务开启：
```bash
02/20 13:14:15 - AISBench - INFO - Found 10 tmp items, run infer task from the last interrupted position
```
续推结束后，会重新所有请求的精度结果并打印，生成结果与[多任务测评](#多任务测评)示例一致。


> ⚠️ 注意：中断续测与失败重测可能改变请求顺序，可能引发结果微小波动。

### 合并子数据集推理
部分数据集会分类成不同的子数据集，在推理时会被划分为多个子任务行推理，例如：[MMLU](../../ais_bench/benchmark/configs/datasets/mmlu/README.md)、[CEVAL](../../ais_bench/benchmark/configs/datasets/ceval/README.md)。AISBench Benchmark支持将存在多个小规模数据集的数据集合并为一个任务进行统一测评。示例如下：
```bash
ais_bench --models vllm_api_general --datasets ceval_gen --merge-ds
```
> ⚠️ 注意：合并模式下将只生成整体结果，子数据集精度不再单独列出。

# 纯模型精度测评
在本地环境加载模型与数据集，通过统一推理流程比对输出与参考答案，评估模型固有准确率。自定义批量大小、序列长度等参数，适用于**Huggingface Transformers**推理框架。
## 测试准备
在执行服务化推理前，需要满足以下条件：

- 可用的模型权重：确保本地已有需测试的模型权重文件，开源权重可从[huggingface社区](https://huggingface.co/models)获取。
- 数据集准备：支持使用[开源数据集](datasets.md#开源数据集)或[自定义数据集](datasets.md#自定义数据集)。建议将开源数据集手动放置在默认目录 ais_bench/datasets/ 下，程序将在任务执行时自动加载数据集文件。
- 服务化模型后端配置：请参考[本地模型后端](models.md#本地模型后端)进行后端配置，详细参数说明见[配置参数说明](models.md#本地模型后端配置参数说明)。

## 主要功能
纯模型精度测评场景下主要功能与服务化精度测评场景相似。
### 多任务测评
参考[服务化精度多任务测评使用方法](#多任务测评)
### 多任务并行测评
参考[服务化精度多任务并行测评使用方法](#多任务并行测评)。
> ⚠️ 注意：纯模型精度测评多任务并行会占用不同GPU单元，并行任务所需的GPU单元应小于等于可使用的GPU总数。
### 中断续测
支持从推理任务中断位置继续推理剩余请求。适用于Benchmark过程意外中断的推理任务失败场景。
1、当用户使用如下命令首次推理时，由于任务异常退出导致的任务中断
```bash
ais_bench --models hf_base_model --datasets gsm8k_gen
```
此时部分推理结果会被保存下来，在[`--work-dir`](cli_args.md#公共参数)生成如下文件内容：

```bash
outputs/default/
└── 20230220_196253     # 任务创建时间对应的输出目录
    ├── configs         # 存储转储的配置文件。如果在同一实验目录下重复执行多个任务，可能会存在多个配置文件
    │    └── 20230220_196253_7234891.py
    ├── logs            # 推理与精度计算阶段的日志文件
    │   └── infer       # 推理阶段日志
    │       └── hf-base-model
    │           └── gsm8k.out
    └── predictions     # 本地模型推理结果，包含每条请求的输入、模型输出以及参考答案（用于精度评估）
        └── hf-base-model
            └── tmp_gsm8k.json   # 推理结果缓存文件，命名格式为：tmp_{数据集名称}.json

```
2、通过`--reuse`参数指定任务时间戳目录续推：
```bash
ais_bench --models hf_base_model --datasets gsm8k_gen --reuse 20230220_196253
```
日志中会打印如下内容，提示续推任务开启：
```bash
02/20 13:14:15 - AISBench - INFO - Found 10 tmp items, run infer task from the last interrupted position
```
续推结束后，会重新所有请求的精度结果并打印，生成结果可参考[多任务测评](#多任务测评)。
### 合并子数据集推理
参考[服务化精度合并子数据集推理使用方法](#合并子数据集推理)。

# FAQ