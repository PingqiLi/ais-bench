# 服务化性能测评指南
## 测试准备
在执行服务化推理前，需要满足以下条件：

- 可访问的服务化模型服务：确保服务进程可在当前环境下直接访问。
- 数据集准备：支持使用[全量数据集](datasets.md#支持数据集类型)。建议将开源数据集手动放置在默认目录 `ais_bench/datasets/` 下，程序将在任务执行时自动加载数据集文件。
- 服务化模型后端配置：请参考[服务化推理后端](models.md#服务化推理后端)进行后端配置，详细参数说明见[配置参数说明](models.md#服务化推理后端配置参数说明)。

## 服务化性能测评快速入门
AISBench Benchmark 提供服务化性能测评能力。针对流式推理场景，通过精确记录每条请求的发送时间、各阶段返回时间及响应内容，系统地评估模型服务在实际部署环境中的响应延迟（如 TTFT、Token间延迟）、吞吐能力（如 QPS、TPUT）、并发处理能力等关键性能指标。

用户可通过配置服务化后端参数，灵活控制请求内容、请求间隔、并发数量等，适配不同评测场景（如低并发延迟敏感型、高并发吞吐优先型等）。测评支持自动化执行并输出结构化结果，便于横向对比不同模型、部署方案、硬件配置下的服务性能差异。
### 测试样例
服务化性能测评仅支持[服务化推理后端](./models.md#服务化推理后端)中的接口类型为`流式接口`的子服务。
在任意路径下执行命令：
  ```bash
    # 命令行界面 (CLI)
    ais_bench --models {模型配置} --datasets {数据集配置} -m perf [OPTIONS]
    #示例：ais_bench --models vllm_api_general --datasets gsm8k_gen -m perf
```
`[OPTIONS]`为ais_bench的可选参数，详细介绍可查看[CLI全量参数说明](cli_args.md)

测评vLLM 服务化后端为例，修改其[`v1/chat/completions`子服务](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py)的模型配置文件
```python
from ais_bench.benchmark.models import VLLMCustomAPIChatStream

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChatStream,
        abbr='vllm-api-stream-chat',
        path="",                    # 指定模型序列化词表文件路径
        model="DeepSeek-R1",        # 指定服务端已加载模型名称
        request_rate = 0,           # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        retry = 2,
        host_ip = "localhost",      # 指定推理服务的IP
        host_port = 8080,           # 指定推理服务的端口
        max_out_len = 512,
        batch_size=1,               # 请求发送的最大并发数
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
修改好配置文件后，执行如下命令启动性能评测：
```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt -m perf
```

性能结果打屏示例如下，具体性能参数的含义请参考[性能测评结果说明](./performance_mertic.md)：

```bash
06/05 20:22:24 - AISBench - INFO - Performance Results of task: vllm-api-stream-chat/gsm8kdataset:

╒══════════════════════════╤═════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════╕
│ Performance Parameters   │ Stage   │ Average          │ Min              │ Max              │ Median           │ P75              │ P90              │ P99              │  N   │
╞══════════════════════════╪═════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════╡
│ E2EL                     │ total   │ 2048.2945  ms    │ 1729.7498 ms     │ 3450.96 ms       │ 2491.8789 ms     │ 2750.85 ms       │ 3184.9186 ms     │ 3424.4354 ms     │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TTFT                     │ total   │ 50.332 ms        │ 50.6244 ms       │ 52.0585 ms       │ 50.3237 ms       │ 50.5872 ms       │ 50.7566 ms       │ 50 .0551 ms      │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TPOT                     │ total   │ 10.6965 ms       │ 10.061 ms        │ 10.8805 ms       │ 10.7495 ms       │ 10.7818 ms       │ 10.808 ms        │ 10.8582 ms       │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ ITL                      │ total   │ 10.6965 ms       │ 7.3583 ms        │ 13.7707 ms       │ 10.7513 ms       │ 10.8009 ms       │ 10.8358 ms       │ 10.9322 ms       │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ InputTokens              │ total   │ 1512.5           │ 1481.0           │ 1566.0           │ 1511.5           │ 1520.25          │ 1536.6           │ 1563.06          │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokens             │ total   │ 287.375          │ 200.0            │ 407.0            │ 280.0            │ 322.75           │ 374.8            │ 403.78           │ 8    │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokenThroughput    │ total   │ 115.9216 token/s │ 107.6555 token/s │ 116.5352 token/s │ 117.6448 token/s │ 118.2426 token/s │ 118.3765 token/s │ 118.6388 token/s │ 8    │
╘══════════════════════════╧═════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════╛
╒══════════════════════════╤═════════╤════════════════════╕
│ Common Metric            │ Stage   │ Value              │
╞══════════════════════════╪═════════╪════════════════════╡
│ Benchmark Duration       │ total   │ 19897.8505 ms      │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Requests           │ total   │ 8                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Failed Requests          │ total   │ 0                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Success Requests         │ total   │ 8                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Concurrency              │ total   │ 0.9972             │
├──────────────────────────┼─────────┼────────────────────┤
│ Max Concurrency          │ total   │ 1                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Request Throughput       │ total   │ 0.4021 req/s       │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Input Tokens       │ total   │ 12100              │
├──────────────────────────┼─────────┼────────────────────┤
│ Prefill Token Throughput │ total   │ 17014.3123 token/s │
├──────────────────────────┼─────────┼────────────────────┤
│ Total generated tokens   │ total   │ 2299               │
├──────────────────────────┼─────────┼────────────────────┤
│ Input Token Throughput   │ total   │ 608.7438 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Output Token Throughput  │ total   │ 115.7835 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Token Throughput   │ total   │ 723.5273 token/s   │
╘══════════════════════════╧═════════╧════════════════════╛

06/05 20:22:24 - AISBench - INFO - Performance Result files locate in outputs/default/20250605_202220/performances/vllm-api-stream-chat.

```

## 主要功能
### 多任务测评
用户可通过`--models`和`--datasets`参数指定多个配置任务，子任务数为`--models`配置任务数和`--datasets`配置任务数的乘积，即一个模型配置和一个数据集配置组成一个子任务，示例：
```bash
ais_bench --models vllm_api_general_stream vllm_api_stream_chat --datasets gsm8k_gen math500_gen_0_shot_cot_chat_prompt
```
上述命令将执行以下4个性能测试任务：
+ [`v1/completions`](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) + [`GSM8K`](../../ais_bench/benchmark/configs/datasets/gsm8k/README.md) 数据集
+ [`v1/completions`](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py) + [`MATH`](../../ais_bench/benchmark/configs/datasets/math/README.md) 数据集
+ [`v1/chat/completions`](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) + [`GSM8K`](../../ais_bench/benchmark/configs/datasets/gsm8k/README.md) 数据集
+ [`v1/chat/completions`](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py) + [`MATH`](../../ais_bench/benchmark/configs/datasets/math/README.md) 数据集

测试完成后，AISBench 会在指定的 [`--work-dir`](cli_args.md#公共参数) 路径下生成如下目录结构：
```bash
outputs/default/
├── 20240220_120000           # 每次实验基于时间戳生成的唯一目录
│   ├── configs               # 自动存储的所有已转储配置文件
│   ├── logs
│   │   └── performance       # 推理阶段的日志文件
│   └── performance           # 性能测评结果
│       ├── vllm-api-general-stream/      # “服务化模型配置”名称，对应 abbr
│       │    ├── gsm8dataset.csv          # 单次请求性能输出（CSV）
│       │    ├── gsm8dataset.json         # 端到端性能输出（JSON）
│       │    ├── gsm8dataset_details.json # 全量打点日志（JSON）
│       │    ├── gsm8dataset_plot.html    # 请求并发可视化报告（HTML）
│       │    └── …                         # 其他对应文件
│       └── vllm-api-stream-chat/
│            ├── …                         # 同上

```
> ⚠️ 注意：在性能测评场景下，由于需要维护并发连续性，无法使用 `--disable-cb` 关闭 Continuous Batch；也无法通过 `--max-num-workers` 进行子任务并行推理（多任务并行在此模式下受限）。

### 自定义序列长度测评
自定义序列长度测评
如果想要针对特定的输入长度分布进行性能测试，可先通过配置[随机合成数据集](./datasets.md#配置随机合成数据集) 来生成固定输入序列长度。具体步骤如下：

1. 配置随机合成数据集:
请参考 [随机合成数据集](./datasets.md#配置随机合成数据集) 文档，设定输入输出序列长度分布规则。

2. 配置后处理参数 `ignore_eos`:
在[服务化模型配置](./models.md#服务化推理后端配置参数说明)的 `generation_kwargs` 中添加 `ignore_eos = True`，以控制请求的最大输出长度（不提前结束）。

3. 启动性能测评:
以 [`v1/chat/completions`](../../ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py) 为例，示例命令如下：
```bash
ais_bench --models  vllm_api_stream_chat --datasets synthetic_gen -m perf 
```
完成后，输出目录结构同[多任务测评](#多任务测评)章节所示，会在 performance/<model_abbr>/<dataset_abbr>* 下生成相应的 CSV/JSON/HTML 文件。
> ⚠️ 注意：
> - 为了保证不同后端 API 的测评维度一致，AISBench Benchmark 会将服务端返回结果先通过 Tokenizer 转换为对应的 Token id，再统计实际生成的 Token 长度。该统计值可能与服务端直接报告的 Token 数略有差异。
> - 部分服务化后端不支持 `ignore_eos` 后处理参数，此时实际输出的 `Token` 数可能无法达到所配置的最大输出长度。
### 最大并发稳态性能测评
「稳态」指在推理服务达到并保持最大并发时的系统运行状态。通常，在请求由启动到达到最大并发这一爬坡阶段，性能数据会受到请求速率（Request Rate）和后端响应效率的共同影响，统计到的性能指标并不能完全反映系统在稳态时的真实能力。
AISBench Benchmark可实时记录每条请求的处理状态和服务化系统的并发状态，在推理任务结束后汇总生成[可视化html报告](./性能测试可视化并发图使用说明.md)。同时用户可通过配置[`--summarizer`参数](./summarizer.md#支持的结果汇总任务)测评系统的稳态性能，示例如下：
```bash
ais_bench --models  vllm_api_stream_chat --datasets synthetic_gen  --summarizer stable_stage  -m perf 
```
由于稳态数据是从全量数据中过滤得到，因此，确保[`--work-dir`](cli_args.md#公共参数)的时间戳文件夹中包含如下文件
```bash
outputs/default/
├── 20240220_120000     # 满足要求的时间戳
│   └── performance       # 性能测评结果
│       └── <model_abbr>  # <model_abbr>与模型配置文件中的models的abbr参数一致
│           ├── <dataset_abbr>_details.json    #全量性能打点结果, <model_abbr>与数据集配置文件中的datasets的abbr参数一致
```
则可以基于原有的性能推理结果，通过修改原有性能测试模式为可视化任务`--mode viz`命令，并添加[`--reuse`](./cli_args.md#公共参数)参数重新生成稳态性能结果：
```bash
ais_bench --models  vllm_api_stream_chat --datasets synthetic_gen  --summarizer stable_stage  --mode perf_viz  --reuse 20240220_120000
```
示例中读取 20240220_120000/performance/vllm-api-stream-chat/syntheticdataset_details.json 中的全量打点数据，重新计算并输出稳态阶段的 CSV/JSON/HTML 等文件。如下所示：

```bash
outputs/default/
├── 20240220_120000     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理阶段的日志文件
│   │   └── performance
│   └── performance       # 性能测评结果
│       └── vllm-api-stream-chat  # 后端模型
│           ├── syntheticdataset.csv     # 新生成，覆盖原来的同名文件
│           ├── syntheticdataset.json    # 新生成，覆盖原来的同名文件
│           ├── syntheticdataset_details.json
│           └── syntheticdataset_plot.html
```
### 固定请求数测评

当集规模过大，只想针数据对部分样本执行性能测试时，可使用 [`--num-prompts`](./cli_args.md#性能测评参数) 参数指定读取的数据条数。示例如下：
```bash
ais_bench --models vllm_api_stream_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt -m perf --num-prompts 1
```
上述命令仅对示例数据集中的第一条记录进行推理并测量性能。
> ⚠️ 注意：当前数据集会按照默认队列顺序依次读取，不支持随机抽样或打乱顺序。
# FAQ