# Quick Start
## Command Meaning
A single or multiple evaluation tasks executed by the AISBench command are defined by a combination of model tasks (single or multiple), dataset tasks (single or multiple), and result presentation tasks (single). Other command-line options of AISBench specify the scenario of the evaluation task (e.g., accuracy evaluation scenario, performance evaluation scenario). Take the following AISBench command as an example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --summarizer example
```
This command does not specify other command-line options, so it defaults to an accuracy evaluation scenario task, where:
- `--models` specifies the model task, i.e., the `vllm_api_general_chat` model task.

- `--datasets` specifies the dataset task, i.e., the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task.

- `--summarizer` specifies the result presentation task, i.e., the `example` result presentation task (if `--summarizer` is not specified, the `example` task is used by default in the accuracy evaluation scenario). It is generally recommended to use the default, so there is no need to specify it in the command line, and subsequent commands will omit it.

## Task Meaning Query (Optional)
The specific information (introduction, usage constraints, etc.) of the selected model task `vllm_api_general_chat`, dataset task `demo_gsm8k_gen_4_shot_cot_chat_prompt`, and result presentation task `example` can be queried from the following links respectively:
- `--models`: 📚 [Service-Oriented Inference Backend](../base_tutorials/all_params/models.md#service-oriented-inference-backend)

- `--datasets`: 📚 [Open Source Datasets](../base_tutorials/all_params/datasets.md#open-source-datasets) → 📚 [Detailed Introduction](https://gitee.com/aisbench/benchmark/tree/master/ais_bench/benchmark/configs/datasets/demo/README_en.md)

- `--summarizer`: 📚 [Result Summary Tasks](../base_tutorials/all_params/summarizer.md)

## Preparations Before Running the Command
- `--models`: To use the `vllm_api_general_chat` model task, you need to prepare an inference service that supports the `v1/chat/completions` sub-service. You can refer to 🔗 [Launching an OpenAI-Compatible Server with VLLM](https://docs.vllm.com.cn/en/latest/getting_started/quickstart.html#openai-compatible-server) to start the inference service.
- `--datasets`: To use the `demo_gsm8k_gen_4_shot_cot_chat_prompt` dataset task, you need to prepare the gsm8k dataset, which can be downloaded from 🔗 [the gsm8k dataset zip package provided by opencompass](http://opencompass.oss-cn-shanghai.aliyuncs.com/datasets/data/gsm8k.zip). Deploy the unzipped `gsm8k/` folder to the `ais_bench/datasets` folder in the root directory of the AISBench evaluation tool.

## Modification of Configuration Files Corresponding to Tasks
Each model task, dataset task, and result presentation task corresponds to a configuration file. You need to modify the content of these configuration files before running the command. The paths of these configuration files can be queried by adding `--search` to the original AISBench command. For example:
```shell
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --search
```
> ⚠️ **Note**: Executing the command with the `search` option will print the absolute paths of the configuration files corresponding to the tasks.

Executing the query command will yield the following results:
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

- The dataset task configuration file `demo_gsm8k_gen_4_shot_cot_chat_prompt.py` in the quick start does not require additional modifications. For an introduction to the content of the dataset task configuration file, please refer to 📚 [Configuring Open Source Datasets](../base_tutorials/all_params/datasets.md#configuring-open-source-datasets).

The model configuration file `vllm_api_general_chat.py` contains configuration content related to model operation and needs to be modified according to the actual situation. The content that needs to be modified in the quick start is marked with comments.
```python
from ais_bench.benchmark.models import VLLMCustomAPIChat

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",                    # Specify the absolute path of the model serialized vocabulary file (configuration is generally not required for accuracy testing scenarios).
        model="DeepSeek-R1",        # Specify the name of the model loaded on the server, configured according to the actual model name pulled by the VLLM inference service (configure as an empty string to get it automatically)
        request_rate = 0,           # Request sending frequency: send 1 request to the server every 1/request_rate seconds; if less than 0.1, all requests are sent at once
        retry = 2,                  # Maximum number of retries per request
        host_ip = "localhost",      # Specify the IP of the inference service
        host_port = 8080,           # Specify the port of the inference service
        max_out_len = 512,          # Maximum number of tokens output by the inference service
        batch_size=1,               # Maximum concurrency for sending requests
        generation_kwargs = dict(   # Model inference parameters shall be configured with reference to the VLLM documentation. The AISBench evaluation tool does not process these parameters, which will be included in the sent request.
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]
```
## Execute the Command
After modifying the configuration files, execute the command to start the service-oriented accuracy evaluation (⚠️ It is recommended to add `--debug` for the first execution, which can print specific logs to the screen, making it easier to handle errors during the process of requesting the inference service):
```bash
# Add --debug to the command line
ais_bench --models vllm_api_general_chat --datasets demo_gsm8k_gen_4_shot_cot_chat_prompt --debug
```
### View Task Execution Details
After executing the AISBench command, the details of the task execution will be continuously saved to the default output path. This output path is indicated in the screen logs during runtime, for example:
```shell
06/28 15:13:26 - AISBench - INFO - Current exp folder: outputs/default/20250628_151326
```

This log indicates that the details of the task execution are saved in `outputs/default/20250628_151326` under the path where the command is executed.
After the command execution is completed, the details of the task execution in `outputs/default/20250628_151326` are as follows:

```shell
20250628_151326/
├── configs # A combined configuration file of the model task, dataset task, and result presentation task
│   └── 20250628_151326_29317.py
├── logs # Logs during execution. If --debug is added to the command, no process logs will be saved to disk (all will be printed directly)
│   ├── eval
│   │   └── vllm-api-general-chat
│   │       └── demo_gsm8k.out # Logs of the accuracy evaluation process based on the inference results in the predictions/ folder
│   └── infer
│       └── vllm-api-general-chat
│           └── demo_gsm8k.out # Inference process logs
├── predictions
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Inference results (all outputs returned by the inference service)
├── results
│   └── vllm-api-general-chat
│       └── demo_gsm8k.json # Raw scores calculated by the accuracy evaluation
└── summary
    ├── summary_20250628_151326.csv # Final accuracy scores presented in table format
    ├── summary_20250628_151326.md # Final accuracy scores presented in markdown format
    └── summary_20250628_151326.txt # Final accuracy scores presented in text format
```
> ⚠️ **Note**: The content of the saved task execution details varies across different evaluation scenarios. Please refer to the guide for specific evaluation scenarios for details.

### Output Results
Since there are only 8 pieces of data, the results will be generated quickly. An example of the result display is as follows:
```bash
dataset                 version  metric   mode  vllm_api_general_chat
----------------------- -------- -------- ----- ----------------------
demo_gsm8k              401e4c   accuracy gen                   62.50
```