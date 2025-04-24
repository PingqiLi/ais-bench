# AISBench benchmark评测工具
## 简介
AISBench benchmark评测工具是基于opencompass开发的评测工具，兼容opencompass的配置文件、数据集、模型后端等具体实现。目前支持评测推理精度。

## 工具安装
AISBench benchmark是由python开发的工具，要求`python == 3.10`
本工具的依赖较多，推荐在conda虚拟环境下安装。
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
因为当前工具的模型后端都是服务化api后端，因此需要额外安装服务化的依赖：
```shell
pip3 install -r requirements/api.txt
```

## 工具卸载
执行命令：
```shell
pip3 uninstall ais_bench_benchmark
```

## 快速入门
在本工具的评测中，每个评估任务由待评估的模型后端和数据集组成，可以通过两种方式来指定模型和数据集：命令行指定模型和数据集以及在配置文件中指定模型和数据集，两种方式二选一。当前工具支持的模型后端主要服务化api，以评测gpu上部署的vllm推理服务为例，请先参考[vllm官方文档/启动服务器样例](https://vllm.hyper.ai/docs/tutorials/vLLM-stepbysteb#%E4%B8%89%E5%90%AF%E5%8A%A8-vllm-%E6%9C%8D%E5%8A%A1%E5%99%A8)在gpu服务器上拉起vllm的推理服务。<br>
### 精度评测场景
#### gsm8k数据集准备
参考[gsm8k数据集说明](ais_bench/benchmark/configs/datasets/gsm8k/README.md)准备数据集，将数据集放在ais_bench/datasets路径下。

#### 命令行指定模型和数据集
命令行方式指定模型和数据集本质上是调用工具内置的.py配置文件指定的，需要先在`ais_bench/benchmark/configs/models/`中预置的模型配置文件中配置好服务化相关参数，以执行vllm_api_general的任务为例，需要在`ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py`中修改配置：

```python
from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        type=VLLMCustomAPI,
        attr='service',
        abbr='vllm-api-general',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        enable_ssl = False,
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
修改好配置文件后，执行如下命令启动评测：
```
ais_bench --models vllm_api_general --datasets gsm8k_gen
```
**注:** --models支持的任务参考[--models支持的模型推理后端](#--models支持的模型推理后端)章节，--datasets 支持的任务参考[--datasets支持的数据集](#--datasets支持的数据集)章节。

#### 配置文件指定模型和数据集
需要先在源码中提供的样例配置文件`ais_bench/configs/`中预置的模型配置文件中配置好服务化相关参数，例如要执行的配置文件是`ais_bench/configs/api_examples/infer_api_vllm_general.py`，需要在此配置文件中修改配置：

```python
from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    # from ais_bench.benchmark.configs.datasets.collections.chat_medium import datasets
    from ais_bench.benchmark.configs.summarizers.medium import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

datasets = [
    *gsm8k_datasets,
]


models = [
    dict(
        type=VLLMCustomAPI,
        attr='service',
        abbr='vllm-api-general',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost", # 指定服务化的 host ip
        host_port = 8080, # 指定服务化的端口
        enable_ssl = False,
        max_out_len=512,
    )
]


infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 concurrent_users=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api_vllm_general/' # 指定落盘文件（执行过程、推理结果等）的落盘文件夹

```
修改好配置文件后，执行如下命令启动评测：
```
ais_bench ais_bench/configs/api_examples/infer_api_vllm_general.py
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
#### 配置随机数据集



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
|--models|指定模型推理后端任务名称（对应ais_bench/benchmark/configs/models路径下一个已经实现的默认模型配置文件），支持传入多个任务名称，支持的任务范围请参考[任务支持范围](#任务支持范围)章节<br>此参数在“命令行指定模型和数据集”方式中必须配置，与“配置文件指定模型和数据集”方式中配置的`config` 参数二选一|--models vllm_api_general|
|--datasets|指定数据集任务名称（对应ais_bench/benchmark/configs/datasets路径下一个已经实现的默认数据集配置文件），支持的任务范围请参考[任务支持范围](#任务支持范围)章节<br>此参数在“命令行指定模型和数据集”方式中必须配置，与“配置文件指定模型和数据集”方式中配置的`config`参数二选一|--datasets gsm8k_gen|
|--summarizer|指定结果总结任务名称（对应ais_bench/benchmark/configs/summarizers路径下一个已经实现的默认模型配置文件），支持的任务范围请参考[任务支持范围](#任务支持范围)章节|--summarizer medium|
|--debug|debug模式开关，配置该参数表示开启，未配置表示关闭，默认未配置|--debug|
|--dry-run|dry run模式（只打屏不实际跑任务）开关，配置该参数表示开启，未配置表示关闭，默认未配置|--dry-run|
|--mode 或 -m|可选["all", "infer", "eval", "viz"]，默认"all"，每个模式如何运行参考[运行模式说明](#运行模式说明)|--mode infer <br>-m all|
|--reuse 或 -r|指定重复使用的工作路径下的文件夹时间戳，如果此可选命令不加参数，默认寻找--work_dir指定的工作路径下最新的时间戳|--reuse <br>-r 20250126_144254|
|--work-dir 或 -w|评测任务的工作路径，用于落盘评测过程中的结果文件，默认outputs/default| --work-dir /path/to/work <br>-w /path/to/work|
|--config-dir|models，datasets和summarizers配置文件所在的文件夹路径， 默认ais_bench/benchmark/configs|--config-dir /xxx/xxx|
|--max-num-workers|并行运行的worker的最大个数，默认1|--max-num-workers 1|
|--max-workers-per-gpu|预留参数，暂不支持使用。<br> 评测需要用到的NPU或GPU数量，默认1。|--max-workers-per-gpu 1|
|--dump-eval-details|是否dump出评测过程细节的开关，配置该参数表示开启，未配置表示关闭，默认未配置|--dump-eval-details|
|--dump-extract-rate|是否dump出评测速度的开关，配置该参数表示开启，未配置表示关闭，默认未配置|--dump-extract-rate|
|--merge-ds|是否合并同类数据集为同一个任务来推理的开关，配置该参数表示开启，未配置表示关闭，默认未配置|--merge-ds|
|--disable-cb|是否关闭continous batch的推理方式，配置该参数表示关闭，未配置表示开启，默认未配置。此参数仅对--models指定为服务化API类型的推理后端时才有效。continous batch开启情况下，--reuse无法继承推理结果进行续推。|--disable-cb|



## 运行模式说明
### all 模式

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

### infer模式

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

### eval模式
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

### viz模式
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

## 预设任务支持范围
本节介绍当前ais_bench评测工具支持的评测任务的预设配置，通过ais_bench命令行指定任务名称，即可执行相应的评测任务。
命令示例如下：
```shell
ais_bench --models vllm_api_general --datasets gsm8k_gen --summarizer medium
```
### --models支持的模型推理后端
--models支持两种后端：服务化推理API后端和本地模型后端，两种后端不能在--models中同时指定
#### 服务化推理API后端
|任务名称|简介|使用前提|支持的prompt格式(字符串格式或对话格式)|对应源码配置文件路径|
| --- | --- | --- | --- | --- |
|vllm_api_general|通过vllm的api访问vllm(0.6+版本)的推理服务化，访问服务链接的 v1/completions子服务|基于支持v1/completions子服务的vllm版本，启动vllm推理服务|字符串格式|[vllm_api_general.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general.py)|
|vllm_api_general_chat|通过vllm的api访问vllm(0.6+版本)的推理服务化，访问服务链接的 v1/chat/completions子服务|基于支持v1/chat/completions子服务的vllm版本，启动vllm推理服务|字符串格式、对话格式|[vllm_api_general_chat.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_general_chat.py)|
|vllm_api_stream_chat|通过vllm的流式api访问vllm(0.6+版本)的推理服务化，访问服务链接的 v1/chat/completions子服务|基于支持v1/chat/completions子服务的vllm版本，启动vllm推理服务|字符串格式、对话格式|[vllm_api_stream_chat.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_stream_chat.py)|
|vllm_api_old|通过vllm的api访问vllm(0.2.6版本)的推理服务化，访问服务链接的 generate子服务|基于支持generate子服务的vllm版本，启动vllm推理服务|字符串格式|[vllm_api_old.py](ais_bench/benchmark/configs/models/vllm_api/vllm_api_old.py)|
|mindie_stream_api_general|通过mindie的流式api访问mindie的推理服务化，访问服务链接的 infer子服务|基于支持infer子服务的mindie版本，启动mindie推理服务|字符串格式|[mindie_stream_api_general.py](ais_bench/benchmark/configs/models/mindie_api/mindie_stream_api_general.py)|
**注意:** 服务化推理测评api默认使用的url为localhost，端口号为8080，实际使用时需要在具体使用的源码配置文件中修改为服务化后端配置的url和端口号。

#### 本地模型后端
|任务名称|简介|使用前提|支持的prompt格式(字符串格式或对话格式)|对应源码配置文件路径|
| --- | --- | --- | --- | --- |
|hf_base_model|huggingface base模型后端|安装好评测工具的基础依赖，在源码配置文件中配置好huggingface模型权重路径(当前不支持自动下载权重)|字符串格式|[hf_base_model](ais_bench/benchmark/configs/models/hf_models/hf_base_model.py)|
|hf_chat_model|huggingface chat模型后端|安装好评测工具的基础依赖，在源码配置文件中配置好huggingface模型权重路径(当前不支持自动下载权重)|对话格式|[hf_chat_model](ais_bench/benchmark/configs/models/hf_models/hf_chat_model.py)|


### --datasets支持的数据集
--datasets 支持的数据集如下，每个数据集包含多种数据集任务，数据集的获取方式和支持的数据集任务请参考对应数据集的README。
|数据集|数据集任务README|
| ---- | ---- |
|GSM8K|[ais_bench/benchmark/configs/datasets/gsm8k/README.md](ais_bench/benchmark/configs/datasets/gsm8k/README.md)|
|MMLU|[ais_bench/benchmark/configs/datasets/mmlu/README.md](ais_bench/benchmark/configs/datasets/mmlu/README.md)|
|BoolQ|[ais_bench/benchmark/configs/datasets/SuperGLUE_BoolQ/README.md](ais_bench/benchmark/configs/datasets/SuperGLUE_BoolQ/README.md)|
|C-Eval|[ais_bench/benchmark/configs/datasets/ceval/README.md](ais_bench/benchmark/configs/datasets/ceval/README.md)|
|MATH|[ais_bench/benchmark/configs/datasets/math/README.md](ais_bench/benchmark/configs/datasets/math/README.md)|
|AIME2024|[ais_bench/benchmark/configs/datasets/aime2024/README.md](ais_bench/benchmark/configs/datasets/aime2024/README.md)|
|GPQA|[ais_bench/benchmark/configs/datasets/gpqa/README.md](ais_bench/benchmark/configs/datasets/gpqa/README.md)|
|DROP|[ais_bench/benchmark/configs/datasets/drop/README.md](ais_bench/benchmark/configs/datasets/drop/README.md)|
|HumanEval|[ais_bench/benchmark/configs/datasets/humaneval/README.md](ais_bench/benchmark/configs/datasets/humaneval/README.md)|
|MMLU-PRO|[ais_bench/benchmark/configs/datasets/mmlu_pro/README.md](ais_bench/benchmark/configs/datasets/mmlu_pro/README.md)|
|AGIEval|[ais_bench/benchmark/configs/datasets/agieval/README.md](ais_bench/benchmark/configs/datasets/agieval/README.md)|
|ARC Challenge Set|[ais_bench/benchmark/configs/datasets/ARC_c/README.md](ais_bench/benchmark/configs/datasets/ARC_c/README.md)|
|ARC Easy Set|[ais_bench/benchmark/configs/datasets/ARC_e/README.md](ais_bench/benchmark/configs/datasets/ARC_e/README.md)|
|LiveCodeBench|[ais_bench/benchmark/configs/datasets/livecodebench/README.md](ais_bench/benchmark/configs/datasets/livecodebench/README.md)|
|HellaSwag|[ais_bench/benchmark/configs/datasets/hellaswag/README.md](ais_bench/benchmark/configs/datasets/hellaswag/README.md)|
|mgsm|[ais_bench/benchmark/configs/datasets/mgsm/README.md](ais_bench/benchmark/configs/datasets/mgsm/README.md)|
|mbpp|[ais_bench/benchmark/configs/datasets/mbpp/README.md](ais_bench/benchmark/configs/datasets/mbpp/README.md)|
|piqa|[ais_bench/benchmark/configs/datasets/piqa/README.md](ais_bench/benchmark/configs/datasets/piqa/README.md)|
|TriviaQA|[ais_bench/benchmark/configs/datasets/triviaqa/README.md](ais_bench/benchmark/configs/datasets/triviaqa/README.md)|
|WinoGrande|[ais_bench/benchmark/configs/datasets/winogrande/README.md](ais_bench/benchmark/configs/datasets/winogrande/README.md)|
|CMMLU|[ais_bench/benchmark/configs/datasets/cmmlu/README.md](ais_bench/benchmark/configs/datasets/cmmlu/README.md)|
|BBH|[ais_bench/benchmark/configs/datasets/bbh/README.md](ais_bench/benchmark/configs/datasets/bbh/README.md)|

### --summarizer支持的结果总结任务
|任务名称|简介|对应源码配置文件路径|
| --- | --- | --- |
|medium|通用结果汇总模板，包含多种基本数据集，默认使用的模板|[medium.py](ais_bench/benchmark/configs/summarizers/medium.py)|
|example|简单的汇总模板，覆盖目前所有支持的数据集|[example.py](ais_bench/benchmark/configs/summarizers/example.py)|

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
