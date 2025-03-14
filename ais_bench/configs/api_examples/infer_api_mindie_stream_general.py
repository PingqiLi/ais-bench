from mmengine.config import read_base
from ais_bench.benchmark.models import MindieStreamApi
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    # from ais_bench.benchmark.configs.datasets.collections.chat_medium import datasets
    from ais_bench.benchmark.configs.summarizers.medium import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets

datasets = [
    *gsm8k_datasets,
]


models = [
    dict(
        type=MindieStreamApi,
        abbr='mindie-stream-api',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost",
        host_port = 8080,
        enable_ssl = False,
        generation_kwargs = dict(
            temperature = 0.5, 
            top_k = 10,
            top_p = 0.95,
            max_new_tokens = 512,
            do_sample = True,
            seed = None,
            repetition_penalty = 1.03,
            details = True,
            typical_p = 0.5,
            watermark = False,
            priority = 5,
            timeout = None,
        )
    )
]


infer = dict(partitioner=dict(type=NaivePartitioner),
             runner=dict(
                 type=LocalAPIRunner,
                 max_num_workers=2,
                 concurrent_users=2,
                 task=dict(type=OpenICLInferTask)), )

work_dir = 'outputs/api-mindie-stream/'