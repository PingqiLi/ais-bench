from mmengine.config import read_base
from ais_bench.benchmark.models import VLLMCustomAPIOld
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    # from ais_bench.benchmark.configs.datasets.collections.chat_medium import datasets
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.drop.drop_gen_a2697c_0shot import drop_datasets

datasets = [
    *drop_datasets,
]


models = [
    dict(
        type=VLLMCustomAPIOld,
        abbr='mindie-vllm-api-drop',
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

work_dir = 'outputs/api-mindie-vllm-drop/'