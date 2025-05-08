from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        attr="service", # local or service
        type=VLLMCustomAPI,
        abbr='vllm-api-general',
        path="",
        model="",
        max_seq_len = 4096,
        request_rate = 0,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost", # 推理服务的IP
        host_port = 8080, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 512, # 最大输出tokens长度
        batch_size=1, # 推理的最大并发数
        generation_kwargs = dict( # 后处理参数参考https://docs.vllm.ai/en/latest/api/inference_params.html#sampling-params 中的Parameters
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]
