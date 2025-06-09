# Python自定义配置文件样例列表

|文件名|简介|
| --- | --- |
|[infer_vllm_api_general.py](../../ais_bench/configs/api_examples/infer_vllm_api_general.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/completions子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_mindie_stream_api_general.py](../../ais_bench/configs/api_examples/infer_mindie_stream_api_general.py)|基于gsm8k数据集使用mindie stream api访问infer子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_vllm_api_old.py](../../ais_bench/configs/api_examples/infer_vllm_api_old.py)|基于gsm8k数据集使用vllm api(0.2.6版本)访问generate子服务进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_vllm_api_general_chat.py](../../ais_bench/configs/api_examples/infer_vllm_api_general_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_vllm_api_stream_chat.py](../../ais_bench/configs/api_examples/infer_vllm_api_stream_chat.py)|基于gsm8k数据集使用vllm api(0.6+版本)访问v1/chat/completions子服务使用流式推理进行评测，prompt格式为对话格式，自定义了数据集路径|
|[infer_hf_base_model.py](../../ais_bench/configs/hf_example/infer_hf_base_model.py)|基于gsm8k数据集使用huggingface base模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|
|[infer_hf_chat_model.py](../../ais_bench/configs/hf_example/infer_hf_chat_model.py)|基于gsm8k数据集使用huggingface chat模型的推理接口进行评测，prompt格式为字符串格式，自定义了数据集路径|

**注**: 上述自定义配置文件如果要评测其他数据集，请从[ais_bench/configs/api_examples/all_dataset_configs.py](../../ais_bench/configs/api_examples/all_dataset_configs.py)导入其他数据集。