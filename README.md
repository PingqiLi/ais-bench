# AISBench benchmark评测工具
## 简介
AISBench benchmark评测工具是基于opencompass开发的评测工具，兼容opencompass的配置文件、数据集、模型后端等具体实现。目前支持评测推理[精度](#精度评测场景)/[性能](#性能评测场景)。

## 工具安装
AISBench benchmark 需要Python 3.10 或 3.11 运行环境。不兼容更低版本（如3.9）或更高版本（如3.12）。
 :tw-1f4cc: 本工具的依赖较多，推荐使用Miniconda管理Python环境以避免冲突。
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

目前只支持源码构建安装，请确保安装环境网络畅通：

```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./
```
安装过程中会自动安装基础依赖。
由于本工具支持多种模型服务框架（如vLLM、Trition等），需额外安装服务化的依赖：
```shell
pip3 install -r requirements/api.txt
```

## 工具卸载
执行命令：
```shell
pip3 uninstall ais_bench_benchmark
```

## 快速入门
在本工具的评测中，每个评估任务由待评估的模型后端和数据集组成，可以通过两种方式来指定模型和数据集：命令行指定模型和数据集以及在配置文件中指定模型和数据集，两种方式二选一。当前工具支持的模型后端主要服务化api，以评测gpu上部署的vllm推理服务为例，请先参考[vllm官方文档/启动服务器样例](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html)在gpu服务器上拉起vllm的推理服务。<br>
### 精度评测场景
#### gsm8k数据集准备
参考[gsm8k数据集说明](ais_bench/benchmark/configs/datasets/gsm8k/README.md)准备数据集，将数据集放在ais_bench/datasets路径下。

#### 命令行指定模型和数据集
通过命令行方式指定模型和数据集时，需基于预置的.py配置文件来配置服务化参数，以执行vllm_api_general的任务为例，需要在[ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)中修改配置：

```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service", # local or service
        type=VLLMCustomAPI, # API的类名称
        abbr='vllm-api-general', # api的唯一标识，多任务时用于区分任务名
        path="", # 模型tokenizer文件路径，通常为权重路径
        model="", # server端模型名称
        max_seq_len = 4096, # 最大输入序列长度
        request_rate = 0, # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        rpm_verbose = False,
        retry = 2, # server端链接错误重试次数
        host_ip = "localhost", # 推理服务的IP
        host_port = 8080, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 512, # 最大输出tokens长度
        batch_size=1, # service请求发送的最大并发数
        generation_kwargs = dict( # 后处理参数参考https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params 中的Parameters
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
```
ais_bench --models vllm_api_general --datasets gsm8k_gen
```
**注:** --models支持的任务参考[--models支持的模型推理后端](#--models支持的模型推理后端)章节，--datasets 支持的任务参考[--datasets支持的数据集](#--datasets支持的数据集)章节。

#### 配置文件指定模型和数据集
需要先在样例配置文件中配置好服务化相关参数，例如要执行的配置文件是[ais_bench/configs/api_examples/infer_vllm_api_general.py](ais_bench/configs/api_examples/infer_vllm_api_general.py)，需要在此配置文件中修改配置：

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str

datasets = [ # all_dataset_configs.py中导入了其他数据集配置，可以将gsm8k_0_shot_cot_str替换为其他一个或多个数据集
    *gsm8k_0_shot_cot_str,
]

models = [
    dict(
        attr="service", # local or service
        type=VLLMCustomAPI, # API的类名称
        abbr='vllm-api-general', # api的唯一标识，多任务时用于区分任务名
        path="/path/to/tokenizer", # 模型tokenizer文件路径，通常为权重路径
        model="DeepSeekR1", # server端模型名称
        max_seq_len = 4096, # 最大输入序列长度
        request_rate = 0, # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        rpm_verbose = False,
        retry = 2, # server端链接错误重试次数
        host_ip = "localhost", # 推理服务的IP
        host_port = 8080, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 512, # 最大输出tokens长度
        batch_size=1, # service请求发送的最大并发数
        generation_kwargs = dict( # 后处理参数参考https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params 中的Parameters
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]

infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api-vllm-general/' # 指定落盘文件（执行过程、推理结果等）的落盘文件夹

```
修改好配置文件后，执行如下命令启动精度评测：
```
ais_bench ais_bench/configs/api_examples/infer_vllm_api_general.py
```

#### 推理过程查看
启动推理过程中可以在{work_dir}/{time_label}/logs/infer/{abbr_name}/gsm8k.out 中查看推理结果，例如执行
```shell
# 命令行指定模型和数据集运行方式
tail -f outputs/default/20250126_165049/logs/infer/vllm-api-general/gsm8k.out

# 配置文件指定模型和数据集运行方式
tail -f outputs/api_vllm_general/20250126_165049/logs/infer/vllm-api-general/gsm8k.out
```
可以看到推理过程。
其中`{work_dir}/{time_label}/`会在工具的打屏中显示

#### 推理结果查看
推理完成后可以在{work_dir}/{time_label}/predictions/{abbr_name}/gsm8k.json 中查看推理结果，例如执行
```shell
# 命令行指定模型和数据集运行方式
vim outputs/default/20250126_165049/predictions/vllm-api-general/gsm8k.json

# 配置文件指定模型和数据集运行方式
vim outputs/api_vllm_general/20250126_165049/predictions/vllm-api-general/gsm8k.json
```
可以看到推理结果

#### 测评结果查看
在{work_dir}/{time_label}/results/{abbr_name}/gsm8k.json中查看评测出的精度，例如执行
```shell
# 命令行指定模型和数据集运行方式
vim outputs/default/20250126_165049/results/vllm-api-general/gsm8k.json

# 配置文件指定模型和数据集运行方式
vim outputs/api_vllm_general/20250126_165049/results/vllm-api-general/gsm8k.json
```
可以得到类似如下结果：
```json
{
    "accuracy": 59.34
}
```

#### 测评结果可视化
评测过程结束后，工具会将markdown格式的结果打印出来，同时会落盘如下三种格式的结果：
```
{work_dir}/{time_label}/summary/summary_{time_label}.txt
{work_dir}/{time_label}/summary/summary_{time_label}.csv
{work_dir}/{time_label}/summary/summary_{time_label}.md
```
例如
```
outputs/api_vllm_general/20250126_165049/summary/summary_20250126_165049.txt
outputs/api_vllm_general/20250126_165049/summary/summary_20250126_165049.csv
outputs/api_vllm_general/20250126_165049/summary/summary_20250126_165049.md
```

### 性能评测场景
AISBench执行性能测评需指定--mode为perf。以随机数据集性能测评场景为例：
#### 配置随机数据集
打开随机数据集配置文件[ais_bench/datasets/synthetic.py](ais_bench/datasets/synthetic.py)，按需求修改带注释的配置项的取值

```py
"""
# 涉及到数值类型配置参数的最大值无特殊说明时，均应小于 2^20 （= 1 M）
#
# StringConfig中的随机生成方法参数说明:
# -------------------------------------------------
# 包含 输入/出 分布配置 "Method" 和 输入/出 长度配置 "Params"
# 格式说明：
# 输入/出 分布名称 -- "Method" : 输入分布类型
# 输入/出 长度配置 "Params" 内各项参数的名称: 说明及取值范围
#
# [Uniform均匀分布] -- "Method" : "uniform"
#   - MinValue: 最小值，范围为 [1, 2^20]
#   - MaxValue: 最大值, 范围为 [1, 2^20], 可等于MinValue
#
# [Gaussian高斯分布] -- "Method" : "gaussian"
#   - Mean    : 平均值, 范围为 [-3.0e38, 3.0e38]，分布中心位置
#   - Var     : 方差, 范围为[0, 3.0e38]，控制数据分散程度
#   - MinValue: 最小值, 范围为 [1, 2^20], 可低于Mean
#   - MaxValue: 最大值, 范围为 [1, 2^20], 可高于Mean, 可等于MinValue
#
# [Zipf齐夫分布] -- "Method" : "zipf"
#   - Alpha   : 形状参数, 范围为(1.0,10.0], 值越大分布越均匀
#   - MinValue: 最小值, 范围为 [1, 2^20]
#   - MaxValue: 最大值, 范围为 [1, 2^20], 需大于MinValue
"""
synthetic_config = {
    "Type":"tokenid",   # [tokenid/string]，生成的随机数据集类型，支持固定长度的随机tokenid，和随机长度的string，两种类型的数据集
    "RequestCount": 10, # 生成的请求条数
    "StringConfig" : {  # string类型的随机数据集的配置相关项，请参考以上注释处："StringConfig中的随机生成方法参数说明"
        "Input" : {     # 每条请求的输入长度
            "Method": "uniform",
            "Params": {"MinValue": 1, "MaxValue": 200}
        },
        "Output" : {    # 每条请求的输出长度
            "Method": "gaussian",
            "Params": {"Mean": 100, "Var": 200, "MinValue": 1, "MaxValue": 100}
        }
    },
    "TokenIdConfig" : { # tokenid类型的随机数据集的配置相关项
        "ModelPath": "ModelPath", # 模型权重路径，必须与模型侧配置文件中的weight_dir/model/path等表示模型或词汇表的路径相同
        "RequestSize": 10 # 每条请求的长度，即每条请求中token id的个数，应小于等于服务端可支持的最大输入序列长度
    }
}
```
<div style="
  background-color:rgb(230, 246, 255);
  border-left: 8px solid #007bff;
  padding: 12px;
  border-radius: 4px;
  display: flex;
  align-items: start;
">
  <!-- 蓝色叹号图标 -->
  <span style="
    color: #007bff;
    font-size: 1.4em;
    line-height: 1;
    margin-right: 8px;
  "></span>
  <div>
    <!-- “说明”标题 -->
    <strong style="color: #007bff; font-size: 1.2em;">说明：</strong>
    <ul style="margin: 0px 0 0 0px; color: #004085;">
<li>随机 tokenid 模式下，tokenid 会被转为字符串发送，因映射非一一对应，服务端还原时可能与原 tokenid 不一致（tokenid值和长度）。</li>
  <li>使用 vllm_api_stream_chat 或 infer_vllm_api_general_chat 等 chat 接口时，服务端会自动添加 chat 模板，导致传给模型的 token 数略微增加（由模板长度决定）。</li>
    </ul>
  </div>
</div>


#### 命令行指定模型和数据集
通过命令行方式指定模型和数据集时，需基于预置的.py配置文件来配置服务化参数，以执行vllm_api_general的任务为例，需要在[ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py)中修改配置：

```python
from ais_bench.benchmark.models import VLLMCustomAPIStream

models = [
    dict(

        attr="service", # local or service
        type=VLLMCustomAPIStream, # API的类名称
        abbr='vllm-api-general-stream', # api的唯一标识，用于区分任务
        model="DeepSeekR1", # server端模型名称
        path="/path/to/tokenizer" # tokenizer文件所在文件夹路径
        max_seq_len = 4096, # 最大输入序列长度
        request_rate = 0, # 请求发送频率，每1/request_rate秒发送1个请求给服务端，小于0.1则一次性发送所有请求
        rpm_verbose = False,
        retry = 2, # server端链接错误重试次数
        host_ip = "localhost", # 推理服务的IP
        host_port = 8080, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 512, # 最大输出tokens长度
        batch_size=1, # 推理的最大并发数
        generation_kwargs = dict( # 后处理参数参考https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params 中的Parameters
            temperature = 0, # 性能测评设置为0关闭后处理
            ignore_eos = True, # 确保服务化输出的token长度稳定为
        )
    )
]

```
修改好配置文件后，执行如下命令启动性能评测：
```
ais_bench --models vllm_api_general_stream --datasets synthetic_gen --mode perf
```
**注:** --models支持的任务参考[--models支持的模型推理后端](#--models支持的模型推理后端)章节中的流式api。

#### 推理过程查看
启动推理过程中可以在{work_dir}/{time_label}/logs/performances/{abbr_name}/syntheticdataset.out 中查看推理结果，例如执行
```shell
# 命令行指定模型和数据集运行方式
tail -f outputs/default/20250424_202220/logs/infer/vllm-api-general-stream/syntheticdataset.out
```
可以看到推理过程。
其中`{work_dir}/{time_label}/`(例如outputs/default/20250424_202220/)会在工具的打屏中显示

#### 性能结果查看
性能结果打屏示例如下，具体性能参数的含义请参考[性能测评结果说明](#性能测评结果说明)：

```bash
04/24 20:22:24 - AISBench - INFO - Performance Results of task: vllm-api-general-stream/syntheticdataset:

╒══════════════════════════╤═════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════════════════╤══════╕
│ Performance Parameters   │ Stage   │ Average          │ Min              │ Max              │ Median           │ P75              │ P90              │ P99              │  N   │
╞══════════════════════════╪═════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════════════════╪══════╡
│ E2EL                     │ total   │ 11457.7972 ms    │ 10808.7498 ms    │ 11645.96 ms      │ 11511.8789 ms    │ 11544.85 ms      │ 11571.9186 ms    │ 11623.4354 ms    │ 4096 │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TTFT                     │ total   │ 501.332 ms       │ 500.6244 ms      │ 529.0585 ms      │ 501.3237 ms      │ 501.5872 ms      │ 501.7566 ms      │ 502.0551 ms      │ 4096 │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ TPOT                     │ total   │ 10.6965 ms       │ 10.061 ms        │ 10.8805 ms       │ 10.7495 ms       │ 10.7818 ms       │ 10.808 ms        │ 10.8582 ms       │ 4096 │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ ITL                      │ total   │ 10.6965 ms       │ 7.3583 ms        │ 13.7707 ms       │ 10.7513 ms       │ 10.8009 ms       │ 10.8358 ms       │ 10.9322 ms       │ 4096 │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ InputTokens              │ total   │ 100.9243         │ 2.0              │ 199.0            │ 102.0            │ 151.0            │ 180.0            │ 197.05           │ 4096 │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokens             │ total   │ 2989.0           │ 2989.0           │ 2989.0           │ 2989.0           │ 2989.0           │ 2989.0           │ 2989.0           │ 4096 │
├──────────────────────────┼─────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────────────────┼──────┤
│ OutputTokenThroughput    │ total   │ 260.9216 token/s │ 256.6555 token/s │ 276.5352 token/s │ 259.6448 token/s │ 261.2426 token/s │ 265.3765 token/s │ 273.6388 token/s │ 4096 │
╘══════════════════════════╧═════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════════════════╧══════╛
╒══════════════════════════╤═════════╤════════════════════╕
│ Common Metric            │ Stage   │ Value              │
╞══════════════════════════╪═════════╪════════════════════╡
│ Benchmark Duration       │ total   │ 474205.8505 ms     │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Requests           │ total   │ 4096               │
├──────────────────────────┼─────────┼────────────────────┤
│ Failed Requests          │ total   │ 0                  │
├──────────────────────────┼─────────┼────────────────────┤
│ Success Requests         │ total   │ 4096               │
├──────────────────────────┼─────────┼────────────────────┤
│ Concurrency              │ total   │ 98.9679            │
├──────────────────────────┼─────────┼────────────────────┤
│ Max Concurrency          │ total   │ 100                │
├──────────────────────────┼─────────┼────────────────────┤
│ Request Throughput       │ total   │ 8.6376 req/s       │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Input Tokens       │ total   │ 413386             │
├──────────────────────────┼─────────┼────────────────────┤
│ Prefill Token Throughput │ total   │ 201.3123 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Total generated tokens   │ total   │ 12242944           │
├──────────────────────────┼─────────┼────────────────────┤
│ Input Token Throughput   │ total   │ 871.7438 token/s   │
├──────────────────────────┼─────────┼────────────────────┤
│ Output Token Throughput  │ total   │ 25817.7835 token/s │
├──────────────────────────┼─────────┼────────────────────┤
│ Total Token Throughput   │ total   │ 26689.5273 token/s │
╘══════════════════════════╧═════════╧════════════════════╛

04/24 20:22:24 - AISBench - INFO - Performance Result files locate in outputs/default/20250424_202220/performances/vllm-api-general-stream.

```
## 完整命令行说明
### 命令格式说明
```shell
ais_bench [OPTIONS]
```
其中[OPTIONS]为ais_bench的可选参数，具体参数如[参数说明](#参数说明)

### 命令行示例
```shell
# 命令行指定模型和数据集
ais_bench --models vllm_api_general --datasets gsm8k_gen
# 配置文件指定模型和数据集
ais_bench ais_bench/configs/api_examples/infer_api_vllm_general.py --debug
```

### 参数说明
|参数|说明|样例|
| ----- | ----- | ---- |
|config|启动用的配置文件路径(.py)，在“配置文件指定模型和数据集”方式中必须配置，与“命令行指定模型和数据集”方式配置--models和--datasets参数二选一，为ais_bench命令行的第一个参数。自定义配置文件可参考[自定义配置文件样例列表](#自定义配置文件样例列表)|ais_bench xxx/yyy.py|
|--models|指定模型推理后端任务名称（对应ais_bench/benchmark/configs/models路径下一个已经实现的默认模型配置文件），支持传入多个任务名称，支持的任务范围请参考[预设任务支持范围](#预设任务支持范围)章节<br>此参数在“命令行指定模型和数据集”方式中必须配置，与“配置文件指定模型和数据集”方式中配置的`config` 参数二选一|--models vllm_api_general|
|--datasets|指定数据集任务名称（对应ais_bench/benchmark/configs/datasets路径下一个已经实现的默认数据集配置文件），支持的任务范围请参考[预设任务支持范围](#预设任务支持范围)章节<br>此参数在“命令行指定模型和数据集”方式中必须配置，与“配置文件指定模型和数据集”方式中配置的`config`参数二选一|--datasets gsm8k_gen|
|--summarizer|指定结果总结任务名称（对应ais_bench/benchmark/configs/summarizers路径下一个已经实现的默认模型配置文件），支持的任务范围请参考[预设任务支持范围](#预设任务支持范围)章节|--summarizer medium|
|--debug|debug模式开关，配置该参数表示开启，未配置表示关闭，默认未配置。debug模式下所有日志将会直接打印在终端|--debug|
|--dry-run|dry run模式（只打屏不实际跑任务）开关，配置该参数表示开启，未配置表示关闭，默认未配置|--dry-run|
|--mode 或 -m|可选["all", "infer", "eval", "viz", "perf", "perf_viz"]，默认"all"，每个模式如何运行参考[运行模式说明](#运行模式说明)|--mode infer <br>-m all|
|--reuse 或 -r|指定重复使用的工作路径下的文件夹时间戳，如果此可选命令不加参数，默认寻找--work_dir指定的工作路径下最新的时间戳。benchmark会加载该时间戳目录下的数据继续执行任务，结合--mode参数值，可用于推理中断续推，或基于已有推理结果执行精度计算、可视化结果打印|--reuse <br>-r 20250126_144254|
|--work-dir 或 -w|评测任务的工作路径，用于落盘评测过程中的结果文件，默认outputs/default| --work-dir /path/to/work <br>-w /path/to/work|
|--config-dir|models，datasets和summarizers配置文件所在的文件夹路径， 默认ais_bench/benchmark/configs|--config-dir /xxx/xxx|
|--max-num-workers|并行运行的任务的最大个数，取值范围[1，CPU核数]， 默认1。continous batch开启时和性能场景下不生效|--max-num-workers 1|
|--max-workers-per-gpu|预留参数，暂不支持使用。<br> 评测需要用到的NPU或GPU数量，默认1。|--max-workers-per-gpu 1|
|--dump-eval-details|是否dump出评测过程细节的开关，配置该参数表示开启，未配置表示关闭，默认未配置|--dump-eval-details|
|--dump-extract-rate|是否dump出评测速度的开关，配置该参数表示开启，未配置表示关闭，默认未配置|--dump-extract-rate|
|--merge-ds|是否合并同类数据集为同一个任务来推理的开关，配置该参数表示开启，未配置表示关闭，默认未配置|--merge-ds|
|--disable-cb|是否关闭continous batch的推理方式，配置该参数表示关闭，未配置表示开启，默认未配置。此参数仅对--models指定为服务化API类型的推理后端时才有效。continous batch开启情况下，会拉起多个进程执行服务化推理任务，单个进程默认最大请求并发数默认为500，可充分利用硬件资源支持大并发(batch_size)场景下的测评。此时--max-num-workers不会生效。|--disable-cb|
|--num-prompts|指定数据集测评条数，需传入正整数，默认情况下对全量数据集进行测评，仅支持服务化性能测评场景|--num-prompts|
|--pressure|是否开启性能压测方式的开关，仅当 --mode perf时有效，配置该参数表示开启，未配置表示关闭，默认未配置|--pressure|

### 常量参数说明
除了命令行参数，部分无需经常修改的常量可在[ais_bench/benchmark/global_consts.py](ais_bench/benchmark/global_consts.py)中直接配置
|参数|说明|
| ----- | ----- |
| WORKERS_NUM | 并行进程数（取值范围 [0, CPU 核数])，默认为 0 时会根据配置的最大并发数自动分配，若单核并发能力不足可适当增大以提升吞吐率。 |

## 运行模式说明
### 精度评测场景
#### all 模式

all模式下评测工具会完整执行一次评测流程：
```mermaid
graph LR;
A[基于给定数据集执行推理] --> B((推理结果));
B --> C[基于推理结果测评]
C --> D((精度数据))
D --> E[基于精度数据汇总呈现]
E --> F((呈现结果))
```

命令示例：
```shell
ais_bench --models vllm_api_general --datasets gsm8k_gen --mode all
```

生成结构目录结构：
```bash
outputs/default/
├── 20250220_120000
├── 20250220_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果
│   └── summary       # 单个实验的汇总评估结果
├── ...
```

#### infer模式

infer模式下评测工具仅会跑出数据集的推理结果：
```mermaid
graph LR;
A[基于给定数据集执行推理] --> B((推理结果));
```
命令示例：
```shell
ais_bench --models vllm_api_general --datasets gsm8k_gen --mode infer
```

生成结构目录结构：
```bash
outputs/default/
├── 20250220_120000
├── 20250220_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 每个任务的推理结果
├── ...
```

#### eval模式
eval模式下评测工具会基于已有的推理结果跑一遍评测流程和结果呈现的流程，需要结合--reuse命令使用：
```mermaid
graph LR;
B((推理结果)) --> C[基于推理结果测评]
C --> D((精度数据))
D --> E[基于精度数据汇总呈现]
E --> F((呈现结果))
```

命令示例：
```shell
ais_bench --models vllm_api_general --datasets gsm8k_gen --mode eval --reuse
```

生成结构目录结构：
```bash
outputs/default/
├── 20250220_120000
├── 20250220_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果 (eval新增)
├── ...
```

#### viz模式
viz模式下评测工具会基于已有的精度数据跑一遍结果呈现的流程，需要结合--reuse命令使用：
```mermaid
graph LR;
D((精度数据)) --> E[基于精度数据汇总呈现]
E --> F((呈现结果))
```

命令示例：
```shell
ais_bench --models vllm_api_general --datasets gsm8k_gen --mode viz --reuse
```

生成结构目录结构：
```bash
outputs/default/
├── 20250220_120000
├── 20250220_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理和评估阶段的日志文件
│   │   ├── eval
│   │   └── infer
│   ├── predictions   # 每个任务的推理结果
│   ├── results       # 每个任务的评估结果
│   └── summary       # 单个实验的汇总评估结果 (viz新增)
├── ...
```
### 性能评测场景
#### perf模式

perf模式下评测工具会完整执行一次性能评测流程并呈现性能结果：
```mermaid
graph LR;
A[基于给定数据集执行推理] --> B((打点数据));
B --> C[基于性能打点结果计算]
C --> D((性能数据))
D --> E[基于性能数据汇总呈现]
E --> F((呈现结果))
```

命令示例：
```shell
ais_bench --models vllm_api_general_stream --datasets synthetic_gen --mode perf
```
**注意** 性能评测场景下--models当前只支持流式的服务化推理API任务，参考[服务化推理API后端](#服务化推理api后端)

生成结构目录结构：
```bash
outputs/default/
├── 20200220_120000
├── 20230220_183030     # 每个实验一个文件夹
│   ├── configs         # 用于记录的已转储的配置文件。如果在同一个实验文件夹中重新运行了不同的实验，可能会保留多个配置
│   ├── logs            # 推理阶段的日志文件
│   │   └── performance
│   └── performance       # 性能测评结果
│       └── vllm-api-general-stream  #后端模型，以mindie_stream_api为例
│           ├── syntheticdataset.csv     #单个推理请求性能输出结果
│           ├── syntheticdataset.json    #端到端性能输出结果
│           ├── syntheticdataset_details.json    #全量性能打点结果
│           └── syntheticdataset_plot.html       #全量请求以及系统实时并发可视化界面
├── ...
```
性能打屏基于syntheticdataset.csv和syntheticdataset.json


#### perf_viz 模式
perf模式下评测工具会完整执行一次性能评测流程：
```mermaid
graph LR;
D((性能数据)) --> E[基于性能数据汇总呈现]
E --> F((呈现结果))
```

命令示例：
```shell
ais_bench --models vllm_api_general_stream --datasets synthetic_gen --mode perf_viz --reuse
```
性能打屏基于最近一个时间戳中的syntheticdataset.csv和syntheticdataset.json文件

## 性能测评结果说明
性能测评结果包括单个推理请求性能输出结果和端到端性能输出结果，参数说明如下：

### 单个推理请求性能输出结果
部分统计指标解释如下所示：
+ P75：以TPOT为例，所有请求的TPOT的75分位。
+ P90：以TPOT为例，所有请求的TPOT的90分位。
+ P99：以TPOT为例，所有请求的TPOT的99分位。
+ E2EL：单个请求的时延
+ TTFT（Time To First Token）:首token时延
+ TPOT（Time Per Output Token）：每个输出token的平均时延，请求粒度，不含首token
+ ITL（Inter-token Latency）：token间时延，不含首token
+ InputTokens：输入token长度
+ OutputTokens：输出token长度
+ OutputTokenThroughput：output吞吐率
+ Tokenizer：tokenizer时间
+ Detokenizer：detokenizer时间

|Performance Parameters|Stage|Average|Max|Min|Median|P75|P90|P99|N|
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
|E2EL|统计此参数的阶段|平均请求时延|最大请求时延|最小请求时延|请求时延中位数|请求时延75分位值|请求时延90分位值|请求时延99分位值|测试数据量，来源于输入参数|
|TTFT|统计此参数的阶段|首个token平均时延|首个token最大时延|首个token最小时延|首个token中位数时延|首个token75分位时延|首个token90分位时延|首个token99分位时延|测试数据量，来源于输入参数|
|TPOT|统计此参数的阶段|Decode阶段平均时延|最大Decode阶段时延|最小Decode阶段时延|Decode阶段中位数时延|75分位Decode阶段时延|90分位每条请求Decode阶段平均时延|99分位Decode阶段时延|测试数据量，来源于输入参数|
|ITL|统计此参数的阶段|token间平均时延|token间最大时延|token间最小时延|token间中位数时延|token间75分位时延|token间90分位时延|token间99分位时延|测试数据量，来源于输入参数|
|InputTokens|统计此参数的阶段|输入token平均长度|最大输入token长度|最小输入token长度|输入token中位数长度|75分位输入token长度|90分位输入token长度|99分位输入token长度|测试数据量，来源于输入参数|
|OutputTokens|统计此参数的阶段|输出token平均长度|最大输出token长度|最小输出token长度|输出token中位数长度|75分位输出token长度|90分位输出token长度|99分位输出token长度|测试数据量，来源于输入参数|
|OutputTokenThroughput|统计此参数的阶段|平均输出吞吐|最大输出吞吐|最小输出吞吐|中位数输出吞吐|输出吞吐75分位|输出吞吐90分位|输出吞吐99分位|测试数据量，来源于输入参数|

### 端到端性能输出结果
|参数|说明|
| ---- | ---- |
|Benchmark Duration|测试总耗时|
|Total Requests|测试数据量|
|Failed Requests|失败请求数据量（包含空和未返回数据的响应）|
|Success Requests|返回请求总数据量（包含非空和空）|
|Concurrency|系统实际平均并发数|
|Max Concurrency|最大并发数即配置并发数|
|Request Throughput|请求吞吐率|
|Total Input Tokens|输入总token数|
|Prefill Token Throughput|prefill吞吐率|
|Total Output Tokens|输出总token数|
|Input Token Throughput|输入吞吐率|
|Output Token Throughput|输出吞吐率|
|Total Token Throughput|总吞吐率|

## 预设任务支持范围
本节介绍当前ais_bench评测工具支持的评测任务的预设配置，通过ais_bench命令行指定任务名称，即可执行相应的评测任务。
命令示例如下：
```shell
ais_bench --models vllm_api_general --datasets gsm8k_gen --summarizer medium
```
### --models支持的模型推理后端
--models支持两种后端：服务化推理API后端和本地模型后端，两种后端不能在--models中同时指定
#### 服务化推理API后端
|任务名称|简介|使用前提|接口类型|支持的prompt格式(字符串格式或对话格式)|对应源码配置文件路径|
| --- | --- | --- | --- | ---- | --- |
|vllm_api_general|通过vllm兼容openai的api访问vllm(0.6+版本)的推理服务化，访问服务链接的 v1/completions子服务|基于支持v1/completions子服务的vllm版本，启动vllm推理服务|文本接口|字符串格式|[vllm_api_general.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)|
|vllm_api_general_stream|通过vllm兼容openai的流式api访问vllm(0.6+版本)的推理服务化，访问服务链接的 v1/completions子服务|基于支持v1/completions子服务的vllm版本，启动vllm推理服务|流式接口|字符串格式|[vllm_api_general_stream.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_stream.py)|
|vllm_api_general_chat|通过vllm兼容openai的api访问vllm(0.6+版本)的推理服务化，访问服务链接的 v1/chat/completions子服务|基于支持v1/chat/completions子服务的vllm版本，启动vllm推理服务|文本接口|字符串格式、对话格式|[vllm_api_general_chat.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py)|
|vllm_api_stream_chat|通过vllm兼容openai的流式api访问vllm(0.6+版本)的推理服务化，访问服务链接的 v1/chat/completions子服务|基于支持v1/chat/completions子服务的vllm版本，启动vllm推理服务|流式接口|字符串格式、对话格式|[vllm_api_stream_chat.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py)|
|vllm_api_old|通过vllm的api访问vllm(0.2.6版本)的推理服务化，访问服务链接的 generate子服务|基于支持generate子服务的vllm版本，启动vllm推理服务|文本接口|字符串格式|[vllm_api_old.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_old.py)|
|mindie_stream_api_general|通过mindie的流式api访问mindie的推理服务化，访问服务链接的 infer子服务|基于支持infer子服务的mindie版本，启动mindie推理服务|流式接口|字符串格式|[mindie_stream_api_general.py](ais_bench/benchmark/configs/models/mindie_api/mindie_stream_api_general.py)|
|triton_api_general|通过triton规定格式的api访问推理服务化，访问服务链接的v2/models/{model name}/generate子服务|启动支持triton api的推理服务|文本接口|字符串格式|[triton_api_general.py](ais_bench/benchmark/configs/models/triton_api/triton_api_general.py)|
|triton_stream_api_general|通过triton规定格式的流式api访问推理服务化，访问服务链接的v2/models/{model name}/generate_stream子服务|启动支持triton api的推理服务|流式接口|字符串格式|[triton_stream_api_general.py](ais_bench/benchmark/configs/models/triton_api/triton_stream_api_general.py)|
|tgi_api_general|通过TGI规定格式的api访问推理服务化，访问服务的generate子服务|启动支持TGI api的推理服务|文本接口|字符串格式|[tgi_api_general](ais_bench/benchmark/configs/models/tgi_api/tgi_api_general.py)|
|tgi_stream_api_general|通过TGI规定格式的流式api访问推理服务化，访问服务的generate_stream子服务|启动支持TGI api的推理服务|流式接口|字符串格式|[tgi_stream_api_general](ais_bench/benchmark/configs/models/tgi_api/tgi_stream_api_general.py)|


**注意:** 服务化推理测评api默认使用的服务IP为localhost，端口号为8080，实际使用时，需在对应配置文件中修改为服务化后端配置的IP和端口号。

#### 本地模型后端
|任务名称|简介|使用前提|支持的prompt格式(字符串格式或对话格式)|对应源码配置文件路径|
| --- | --- | --- | --- | --- |
|hf_base_model|huggingface base模型后端|安装好评测工具的基础依赖，在源码配置文件中配置好huggingface模型权重路径(当前不支持自动下载权重)|字符串格式|[hf_base_model](ais_bench/benchmark/configs/models/hf_models/hf_base_model.py)|
|hf_chat_model|huggingface chat模型后端|安装好评测工具的基础依赖，在源码配置文件中配置好huggingface模型权重路径(当前不支持自动下载权重)|对话格式|[hf_chat_model](ais_bench/benchmark/configs/models/hf_models/hf_chat_model.py)|


### --datasets支持的数据集
--datasets 支持的数据集如下，每个数据集包含多种数据集任务，数据集的获取方式和支持的数据集任务请参考对应数据集的README。[服务化推理API后端](#服务化推理api后端)中带有`chat`的接口，可以适配所有数据集配置文件，不带`chat`的接口，仅适用于文件名中带有`str`的数据集配置文件，即字符串格式配置文件。
|数据集|数据集任务README|
| ---- | ---- |
|AGIEval|[ais_bench/benchmark/configs/datasets/agieval/README.md](ais_bench/benchmark/configs/datasets/agieval/README.md)|
|AIME2024|[ais_bench/benchmark/configs/datasets/aime2024/README.md](ais_bench/benchmark/configs/datasets/aime2024/README.md)|
|ARC Challenge Set|[ais_bench/benchmark/configs/datasets/ARC_c/README.md](ais_bench/benchmark/configs/datasets/ARC_c/README.md)|
|ARC Easy Set|[ais_bench/benchmark/configs/datasets/ARC_e/README.md](ais_bench/benchmark/configs/datasets/ARC_e/README.md)|
|BBH|[ais_bench/benchmark/configs/datasets/bbh/README.md](ais_bench/benchmark/configs/datasets/bbh/README.md)|
|BoolQ|[ais_bench/benchmark/configs/datasets/SuperGLUE_BoolQ/README.md](ais_bench/benchmark/configs/datasets/SuperGLUE_BoolQ/README.md)|
|CMMLU|[ais_bench/benchmark/configs/datasets/cmmlu/README.md](ais_bench/benchmark/configs/datasets/cmmlu/README.md)|
|C-Eval|[ais_bench/benchmark/configs/datasets/ceval/README.md](ais_bench/benchmark/configs/datasets/ceval/README.md)|
|DROP|[ais_bench/benchmark/configs/datasets/drop/README.md](ais_bench/benchmark/configs/datasets/drop/README.md)|
|GPQA|[ais_bench/benchmark/configs/datasets/gpqa/README.md](ais_bench/benchmark/configs/datasets/gpqa/README.md)|
|GSM8K|[ais_bench/benchmark/configs/datasets/gsm8k/README.md](ais_bench/benchmark/configs/datasets/gsm8k/README.md)|
|HellaSwag|[ais_bench/benchmark/configs/datasets/hellaswag/README.md](ais_bench/benchmark/configs/datasets/hellaswag/README.md)|
|HumanEval|[ais_bench/benchmark/configs/datasets/humaneval/README.md](ais_bench/benchmark/configs/datasets/humaneval/README.md)|
|HumanEval-X|[ais_bench/benchmark/configs/datasets/humanevalx/README.md](ais_bench/benchmark/configs/datasets/humanevalx/README.md)|
|IFEval|[ais_bench/benchmark/configs/datasets/ifeval/README.md](ais_bench/benchmark/configs/datasets/ifeval/README.md)|
|LiveCodeBench|[ais_bench/benchmark/configs/datasets/livecodebench/README.md](ais_bench/benchmark/configs/datasets/livecodebench/README.md)|
|MATH|[ais_bench/benchmark/configs/datasets/math/README.md](ais_bench/benchmark/configs/datasets/math/README.md)|
|MMLU|[ais_bench/benchmark/configs/datasets/mmlu/README.md](ais_bench/benchmark/configs/datasets/mmlu/README.md)|
|MMLU-PRO|[ais_bench/benchmark/configs/datasets/mmlu_pro/README.md](ais_bench/benchmark/configs/datasets/mmlu_pro/README.md)|
|mbpp|[ais_bench/benchmark/configs/datasets/mbpp/README.md](ais_bench/benchmark/configs/datasets/mbpp/README.md)|
|mgsm|[ais_bench/benchmark/configs/datasets/mgsm/README.md](ais_bench/benchmark/configs/datasets/mgsm/README.md)|
|piqa|[ais_bench/benchmark/configs/datasets/piqa/README.md](ais_bench/benchmark/configs/datasets/piqa/README.md)|
|RACE|[ais_bench/benchmark/configs/datasets/race/README.md](ais_bench/benchmark/configs/datasets/race/README.md)|
|TriviaQA|[ais_bench/benchmark/configs/datasets/triviaqa/README.md](ais_bench/benchmark/configs/datasets/triviaqa/README.md)|
|WinoGrande|[ais_bench/benchmark/configs/datasets/winogrande/README.md](ais_bench/benchmark/configs/datasets/winogrande/README.md)|

### --summarizer支持的结果总结任务
|任务名称|简介|对应源码配置文件路径|
| --- | --- | --- |
|medium|通用精度测评结果汇总模板，呈现多种基本数据集，默认使用的模板|[medium.py](ais_bench/benchmark/configs/summarizers/medium.py)|
|example|简单精度测评结果汇总模板，覆盖目前所有支持的数据集|[example.py](ais_bench/benchmark/configs/summarizers/example.py)|
|default_perf|性能测评结果的汇总模板，汇总全量请求的性能数据，支持通过default_perf.py文件手动配置性能测评结果的统计值|[default_perf.py](ais_bench/benchmark/configs/summarizers/perf/default_perf.py)|
|stable_stage|性能测评结果的汇总模板，汇总稳定状态（系统实际并发达到配置最大并发）请求的性能数据，支持通过stable_stage.py文件手动配置性能测评结果的统计值|[stable_stage.py](ais_bench/benchmark/configs/summarizers/perf/stable_stage.py)|

## 自定义配置文件样例列表
|文件名|简介|
| --- | --- |
|[infer_vllm_api_general.py](ais_bench/configs/api_examples/infer_vllm_api_general.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/completions子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_mindie_stream_api_general.py](ais_bench/configs/api_examples/infer_mindie_stream_api_general.py)|基于gsm8k数据集使用mindie stream api访问infer子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_vllm_api_old.py](ais_bench/configs/api_examples/infer_vllm_api_old.py)|基于gsm8k数据集使用vllm api(0.2.6版本)访问generate子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_vllm_api_general_chat.py](ais_bench/configs/api_examples/infer_vllm_api_general_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_vllm_api_stream_chat.py](ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务使用流式推理进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_hf_base_model.py](ais_bench/configs/hf_example/infer_hf_base_model.py)|基于gsm8k数据集使用huggingface base模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_hf_chat_model.py](ais_bench/configs/hf_example/infer_hf_chat_model.py)|基于gsm8k数据集使用huggingface chat模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|

**注**: 上述自定义配置文件如果要评测其他数据集，请从[ais_bench/configs/api_examples/all_dataset_configs.py](ais_bench/configs/api_examples/all_dataset_configs.py)导入其他数据集。


## 其他特性
### 自定义数据集
参考文档[自定义数据集使用说明](doc/自定义数据集使用说明.md)
