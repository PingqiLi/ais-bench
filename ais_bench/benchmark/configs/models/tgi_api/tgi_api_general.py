from ais_bench.benchmark.models import TGICustomAPI

models = [
    dict(
        attr="service", # local or service
        type=TGICustomAPI,
        path="",
        abbr='tgi-api-general',
        max_seq_len = 4096,
        request_rate = 0,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost", # 推理服务的IP
        host_port = 8080, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 512, # 最大输出tokens长度
        batch_size=1, # 推理的最大并发数
        generation_kwargs = dict( # 后处理参数参考huggingface.github.io/text-generation-inference/
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
            do_sample = True,
            repetition_penalty = 1.03,
        )
    )
]
