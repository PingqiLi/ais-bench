from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    # from ais_bench.benchmark.configs.datasets.collections.chat_medium import datasets
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.math.math_prm800k_500_5shot_cot_gen import math_datasets

datasets = [
    *math_datasets,
]


models = [
    dict(
        type=VLLMCustomAPI,
        abbr='vllm-api-gpu-math500',
        max_seq_len=4096,
        query_per_second=1,
        rpm_verbose=False,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        enable_ssl=False,
        generation_kwargs=dict(
            temperature=0,
            seed=1,
        )
    )
]


infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 concurrent_users=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api-vllm-gpu-math/'