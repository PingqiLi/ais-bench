<div align="center">
  <br />
  <br />

  # **AISBench Benchmark Tool**
  #### A Testing Benchmark Tool for the Artificial Intelligence Field
  <!-- Use a separator line instead of a background -->
  ---

[![Release](https://img.shields.io/badge/dynamic/json?logo=gitee&label=release&logoColor=red&color=green&query=$.name&url=https://gitee.com/api/v5/repos/aisbench/benchmark/releases/latest)](https://gitee.com/aisbench/benchmark/releases)<br>
[![Gitee Forks](https://img.shields.io/badge/dynamic/json?logo=forgejo&label=forks&logoColor=blue&color=blue&query=$.forks_count&url=https://gitee.com/api/v5/repos/aisbench/benchmark?)](https://gitee.com/aisbench/benchmark/members)
[![Stars](https://img.shields.io/badge/dynamic/json?logo=ReverbNation&label=stars&logoColor=yellow&color=yellow&query=$.stargazers_count&url=https://gitee.com/api/v5/repos/aisbench/benchmark)](https://gitee.com/aisbench/benchmark/stargazers)
[![Gitee Issues](https://img.shields.io/badge/dynamic/json?logo=gitee&label=issues&logoColor=red&color=red&query=$.open_issues_count&url=https://gitee.com/api/v5/repos/aisbench/benchmark)](https://gitee.com/aisbench/benchmark/issues)<br>
[![License](https://img.shields.io/badge/license-Apache--2.0-red?logo=apache)](https://www.apache.org/licenses/LICENSE-2.0)
<br><br>
[🌐 Official Website](https://www.aisbench.com) |
[📖 Tool Documentation](https://ais-bench-benchmark.readthedocs.io/en/latest/) |
[🔥 Latest Updates](#-latest-updates)|
[🤔 Report Issues](https://gitee.com/aisbench/benchmark/issues/new/choose)
<br><br>[简体中文](README.md) | English
</div>

> ❗<span style="color: red;"><b>Important</b></span>
>
> **⭐️Star this project** to get the latest updates of AISBench Benchmark Tool in real time!

## 🔥 Latest Updates
- **\[2025.9.08\]** Support for 📚[Simulating Real Business Traffic](https://ais-bench-benchmark.readthedocs.io/en/latest/advanced_tutorials/rps_distribution.html): By controlling fluctuations in request sending rates, perceive the performance evaluation results of service deployment in simulated real-world scenarios! 🔥🔥🔥

- **\[2025.8.28\]** Support for 📚[Multiple Independent Repeated Inference Accuracy Scenarios](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id12), calculating accuracy metrics across different dimensions such as pass@k/cons@k/avg@n! 🔬🔬🔬

- **\[2025.8.19\]**
  - Added a dedicated model configuration for Function Call: [vllm_api_function_call_chat](ais_bench/benchmark/configs/models/vllm_api/vllm_api_function_call_chat.py), supporting [BFCL Function Calling Capability Evaluation](ais_bench/benchmark/configs/datasets/BFCL/README_en.md) 🔥🔥🔥
  - Provided [Performance Test Specification Documentation](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/scenes_intro/performance_benchmark.html#id25) supported by the tool, optimizing memory usage and performance calculation of the tool in inference cluster scenarios. For the maximum specification scenario (250K requests, input/output tokens: 4K/4K), memory usage is reduced by 60% (now less than 64GB), and performance result calculation efficiency is improved by 20x. 🚀🚀🚀

- **\[2025.7.15\]**
  - Supported service deployment performance evaluation and visualization for multi-turn dialogue datasets such as [sharegpt](ais_bench/benchmark/configs/datasets/sharegpt/README_en.md) and [mtbench](ais_bench/benchmark/configs/datasets/mtbench/README_en.md). See 📚[Multi-Turn Dialogue Evaluation Guide](https://ais-bench-benchmark.readthedocs.io/en/latest/advanced_tutorials/multiturn_benchmark.html) for evaluation methods! 🔥🔥🔥
  - Enabled the use of [custom datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/advanced_tutorials/custom_dataset.html) in performance evaluation scenarios, supporting the specification of maximum output length at the request granularity! 🔥🔥🔥

- **\[2025.6.19\]** Support for 📚[Performance Evaluation Result Visualization](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/results_intro/performance_visualization.html) to help locate performance bottlenecks of inference services! 🔥🔥🔥

- **\[2025.6.12\]** Supported accuracy and performance evaluation for multimodal datasets including [textvqa](ais_bench/benchmark/configs/datasets/textvqa/README_en.md), [videobench](ais_bench/benchmark/configs/datasets/videobench/README_en.md), and [vocalsound](ais_bench/benchmark/configs/datasets/vocalsound/README_en.md)! 🔥🔥🔥

- **\[2025.6.6\]** AISBench supports steady-state performance evaluation to obtain the true optimal performance of the system. Refer to 📚 [Service Deployment Steady-State Performance Test](doc/users_guide/stable_stage.md) to get started quickly! 🔥🔥🔥

- **\[2025.5.16\]** Supported performance evaluation for high concurrency service deployment (up to 30,000+ concurrent requests). 📚 [Performance Metrics](doc/users_guide/performance_metric.md) are aligned with 🔗 [vllm benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks). See 📚 [Service Deployment Performance Evaluation Guide](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/scenes_intro/performance_benchmark.html) for details! 🔥🔥🔥

- **\[2025.4.30\]** Accuracy evaluation supports resuming from breakpoints and re-evaluating failed cases, significantly improving the robustness of accuracy evaluation. Refer to 📚 [Resume from Interruption & Re-evaluate Failed Cases](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id10) to get started quickly! 🔥🔥🔥

- **\[2025.4.15\]** Optimized the request sending method from fixed-batch to continuous batch mode, significantly improving accuracy evaluation efficiency! 🔥🔥🔥

- **\[2025.4.12\]** Supported merging all multi-file datasets (such as MMLU, Ceval) into a single dataset task for accuracy evaluation. See 📚 [Merge Multi-File Datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/scenes_intro/accuracy_benchmark.html#id11) for details! 🔥🔥🔥


## 🌏 Introduction
AISBench Benchmark is a model evaluation tool built based on [OpenCompass](https://github.com/open-compass/opencompass). It is compatible with OpenCompass’s configuration system, dataset structure, and model backend implementation, and on this basis, extends support for service-deployed models.

Currently, AISBench supports evaluation scenarios for two major types of inference tasks:

🔍 [Accuracy Evaluation](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/scenes_intro/home.html#id2): Supports accuracy verification of service-deployed models and local models on various question-answering and reasoning benchmark datasets.

🚀 [Performance Evaluation](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/scenes_intro/home.html#id5): Supports latency and throughput evaluation of service-deployed models, as well as extreme performance testing under stress test scenarios.


## 🛠️ Tool Installation
✅ Environment Requirements

**Python Version**: Only Python **3.10** or **3.11** is supported.

Python 3.9 and below are not supported, nor are versions 3.12 and above.

**It is recommended to use Conda for environment management** to avoid dependency conflicts:
```shell
conda create --name ais_bench python=3.10 -y
conda activate ais_bench
```

📦 Installation Method (Source Code Installation)

AISBench currently only provides source code installation. Ensure the installation environment has internet access:
```shell
git clone https://gitee.com/aisbench/benchmark.git
cd benchmark/
pip3 install -e ./ --use-pep517
```
This command will automatically install core dependencies.
Execute `ais_bench -h`. If the help information for all command-line options of the AISBench evaluation tool is printed, the installation is successful.

⚙️ Service Deployment Framework Support (Optional)

If you need to evaluate service-deployed models (such as vLLM, Triton, etc.), install additional dependencies:
```shell
pip3 install -r requirements/api.txt
pip3 install -r requirements/extra.txt
```

🔗 Berkeley Function Calling Leaderboard (BFCL) Evaluation Support
```shell
pip3 install -r requirements/bfcl_dependencies.txt --no-deps
```

**Important Note**: Since `bfcl_eval` automatically installs the `pathlib` library (which is already built into Python 3.5+ environments), use the `--no-deps` parameter to skip automatic installation of additional dependencies to avoid version conflicts.

For further configuration or to initiate evaluation tasks using CLI or Python scripts, refer to the [Quick Start Guide](#quick-start).


## ❌ Tool Uninstallation
To uninstall AISBench Benchmark, execute the following command:
```shell
pip3 uninstall ais_bench_benchmark
```


## 🚀 Quick Start
### Command Meaning
A single or multiple evaluation tasks executed by an AISBench command are defined by a combination of model tasks (single or multiple), dataset tasks (single or multiple), and result presentation tasks (single). Other command-line options of AISBench specify the scenario of the evaluation task (accuracy evaluation scenario, performance evaluation scenario, etc.). Take the following AISBench command as an example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```
This command does not specify other command-line options, so it defaults to an accuracy evaluation task, where:
- `--models` specifies the model task: the `vllm_api_general_chat` model task.
- `--datasets` specifies the dataset task: the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task.
- `--summarizer` specifies the result presentation task: the `example` result presentation task (if `--summarizer` is not specified, the `example` task is used by default for accuracy evaluation scenarios). It is generally used as default and does not need to be specified in the command line (subsequent commands will omit this option).


### Task Meaning Query (Optional)
Detailed information (introduction, usage constraints, etc.) about the selected model task (`vllm_api_general_chat`), dataset task (`demo_gsm8k_gen_4_shot_cot_chat_prompt`), and result presentation task (`example`) can be queried from the following links:
- `--models`: 📚 [Service Deployment Inference Backend](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/models.html#id2)
- `--datasets`: 📚 [Open-Source Datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/datasets.html#id3) → 📚 [Detailed Introduction](ais_bench/benchmark/configs/datasets/demo/README_en.md)
- `--summarizer`: 📚 [Result Summary Tasks](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/summarizer.html)


### Preparations Before Running the Command
- `--models`: To use the `vllm_api_general_chat` model task, prepare an inference service that supports the `v1/chat/completions` sub-service. Refer to 🔗 [Start an OpenAI-Compatible Server with VLLM](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server) to launch the inference service.
- `--datasets`: To use the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task, prepare the GSM8K dataset. Download it from 🔗 [GSM8K Dataset Compressed Package Provided by OpenCompass](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip). Deploy the unzipped `gsm8k/` folder to the `ais_bench/datasets` folder under the root path of the AISBench evaluation tool.


### Modify Configuration Files for Corresponding Tasks
Each model task, dataset task, and result presentation task corresponds to a configuration file. These files need to be modified before running the command. The paths of these configuration files can be queried by adding `--search` to the original AISBench command. For example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
> ⚠️ **Note**: Executing the command with the `--search` option will print the absolute paths of the configuration files corresponding to the tasks.

Executing the query command will yield results similar to the following:
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

- The dataset task configuration file `demo_gsm8k_gen_4_shot_cot_chat_prompt.py` in the quick start does not require additional modifications. For an introduction to the content of dataset task configuration files, refer to 📚 [Configure Open-Source Datasets](https://ais-bench-benchmark.readthedocs.io/en/latest/base_tutorials/all_params/datasets.html#id6).

The model configuration file `vllm_api_general_chat.py` contains configuration content related to model operation and needs to be modified according to actual conditions. Modifications required for the quick start are marked with comments:
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",
        model="DeepSeek-R1",        # Specify the name of the model loaded on the server; configure it according to the actual model name pulled by the VLLM inference service (leave it empty to auto-detect)
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",      # Specify the IP address of the inference service
        host_port = 8080,           # Specify the port of the inference service
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


### Execute the Command
After modifying the configuration files, run the command to start the service-based accuracy evaluation (⚠️ For the first execution, it is recommended to add `--debug` to print detailed logs to the screen, which makes it easier to troubleshoot errors that occur when requesting the inference service):
```bash
# Add --debug to the command line
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
```


#### View Task Execution Details
After running the AISBench command, details of the task execution will be continuously saved to the default output path. This output path is indicated in the logs printed to the screen during execution. For example:
```shell
06/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250628_151326
```

This log indicates that the task execution details are saved in `outputs/default/20250628_151326` (under the directory where the command was executed).
After the command completes, the task execution details in `outputs/default/20250628_151326` are structured as follows:

```shell
20250628_151326/
├── configs # Combined configuration file for model tasks, dataset tasks, and result presentation tasks
│   └── 20250628_151326_29317.py
├── logs # Execution logs; if --debug is added to the command, no intermediate logs are saved to disk (all are printed directly to the screen)
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # Logs of the accuracy evaluation process based on inference results in the predictions/ folder
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # Logs of the inference process
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Inference results (all outputs returned by the inference service)
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Raw scores calculated from the accuracy evaluation
└── summary
    ├── summary_20250628_151326.csv # Final accuracy scores (in table format)
    ├── summary_20250628_151326.md # Final accuracy scores (in Markdown format)
    └── summary_20250628_151326.txt # Final accuracy scores (in text format)
```
> ⚠️ **Note**: The content of saved task execution details varies across different evaluation scenarios. For specifics, please refer to the guide for the respective evaluation scenario.


#### Output Results
Since there are only 8 data entries, results will be generated quickly. An example of the output is shown below:
```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```

For more tutorials, please refer to our 👉[Documentation](https://ais-bench-benchmark.readthedocs.io/en/latest/)


## 🔜 Coming Soon
- [ ] **\[2025.10\]** Complete a full refactoring of AISBench to support plug-and-play integration of cutting-edge testing benchmarks within the AISBench framework, addressing the increasingly complex and diverse testing tasks in the industry; while significantly improving usability.
- [ ] **\[2025.11\]** Provide industry-leading multimodal evaluation capabilities.
- [ ] **\[2025.12\]** Provide evaluation capabilities for mainstream industry Agents.
- [x] **\[2025.9\]** Support simulating real task traffic.
- [x] **\[2025.8\]** Add support for performance evaluation of multi-turn dialogue datasets such as ShareGPT and BFCL.
- [x] **\[2025.8\]** Optimize the calculation efficiency of the eval phase in performance evaluation, reduce the tool’s memory usage, and supplement specifications for tool usage.
- [x] **\[2025.7\]** Enable the use of custom datasets in performance evaluation scenarios, supporting the definition of maximum output length limits for individual data entries.


## 🤝 Acknowledgements
- The code of this project is developed based on 🔗 [OpenCompass](https://github.com/open-compass/opencompass) with extensions.
- Some datasets and prompt implementations in this project are modified from [simple-evals](https://github.com/openai/simple-evals).
- The performance metrics tracked in this project’s code are aligned with [VLLM Benchmark](https://github.com/vllm-project/vllm/tree/main/benchmarks).
- The BFCL function calling capability evaluation feature of this project is implemented based on the [Berkeley Function Calling Leaderboard (BFCL)](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard).


<p align="right"><a href="#top">🔝Back to top</a></p>
