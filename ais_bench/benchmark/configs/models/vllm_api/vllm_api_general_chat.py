from ais_bench.benchmark.models import VLLMCustomAPIChat
from ais_bench.benchmark.utils.model_postprocessors import extract_non_reasoning_content

models = [
    dict(
        attr="service",
        type=VLLMCustomAPIChat,
        abbr='vllm-api-general-chat',
        path="",
        model="",
        request_rate = 0,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        max_out_len = 32000,
        batch_size=16,
        trust_remote_code=False,
        generation_kwargs = dict(
            temperature = 1,
            top_k = 1,
            top_p = 1
        ),
        pred_postprocessor=dict(type=extract_non_reasoning_content)
    )
]
