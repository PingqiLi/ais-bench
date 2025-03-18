from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPI
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    # from ais_bench.benchmark.configs.datasets.collections.chat_medium import datasets
    from ais_bench.benchmark.configs.summarizers.medium import summarizer
    from ais_bench.benchmark.configs.datasets.humaneval.humaneval_gen_0_shot import humaneval_datasets

datasets = [
    *humaneval_datasets,
]


models = [
    dict(
        type=VLLMCustomAPI,
        abbr='vllm-api-gpu-humaneval',
        max_seq_len=4096,
        query_per_second=1,
        rpm_verbose=False,
        retry=2,
        host_ip="localhost",
        host_port=8080,
        enable_ssl=False,
        max_out_len=32768,
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

work_dir = 'outputs/api-vllm-gpu-humaneval/'