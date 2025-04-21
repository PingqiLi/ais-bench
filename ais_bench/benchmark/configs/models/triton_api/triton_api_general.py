from ais_bench.benchmark.models import TritonCustomAPI

models = [
    dict(
        attr="service", # local or service
        type=TritonCustomAPI,
        abbr='triton-api-general',
        model_name='qwen',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost", # 推理服务的IP
        host_port = 8080, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 512, # 最大输出tokens长度
        generation_kwargs = dict( # 后处理参数参考https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/protocol/extension_generate.html
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            do_sample = True,
            repetition_penalty = 1.03,
        )
    )
]
