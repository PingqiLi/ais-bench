from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import MTBenchDataset, MTBenchEvaluator, math_postprocess_v2


mtbench_reader_cfg = dict(
    input_columns=['human'],
    output_column='gpt'
)


mtbench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={'type': 'conversations', 'prompt': "human"}
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

mtbench_eval_cfg = dict(
    evaluator=dict(type=MTBenchEvaluator)
)

mtbench_datasets = [
    dict(
        abbr='mtbench',
        type=MTBenchDataset,
        path='ais_bench/datasets/mtbench/question.jsonl', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=mtbench_reader_cfg,
        infer_cfg=mtbench_infer_cfg,
        eval_cfg=mtbench_eval_cfg
    )
]