from mmengine.config import read_base
from ais_bench.benchmark.models import MindieStreamApi
from ais_bench.benchmark.partitioners import NaivePartitioner
from ais_bench.benchmark.runners.local_api import LocalAPIRunner
from ais_bench.benchmark.tasks import OpenICLInferTask

with read_base():
    from ais_bench.benchmark.configs.summarizers.example import summarizer
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str

datasets = [ # all_dataset_configs.py中导入了其他数据集配置，可以将gsm8k_0_shot_cot_str替换为其他一个或多个数据集
    *gsm8k_0_shot_cot_str,
]


models = [
    dict(
        type=MindieStreamApi,
        path='', # 模型tokenizer词表文件路径
        abbr='mindie-stream-api-general',
        max_seq_len = 4096,
        query_per_second = 1,
        rpm_verbose = False,
        retry = 2,
        host_ip = "localhost", # 推理服务的IP
        host_port = 8080, # 推理服务的端口
        enable_ssl = False,
        max_out_len = 512, # 最大输出tokens长度
        generation_kwargs = dict( # 后处理参数参考https://www.hiascend.com/document/detail/zh/mindie/100/mindieservice/servicedev/mindie_service0090.html 中的parameters的子参数
            temperature = 0.5,
            top_k = 10,
            top_p = 0.95,
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

work_dir = 'outputs/api-mindie-stream/' # 工作路径