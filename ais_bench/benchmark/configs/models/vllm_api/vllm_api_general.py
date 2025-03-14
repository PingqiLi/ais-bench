from ais_bench.benchmark.models import VLLMCustomAPI

models = [
    dict(
        type=VLLMCustomAPI,
        abbr='vllm-api-general',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        enable_ssl = False,
        max_out_len = 512,
        generation_kwargs = dict(
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            seed = None,
            repetition_penalty = 1.03,
        )
    )
]
