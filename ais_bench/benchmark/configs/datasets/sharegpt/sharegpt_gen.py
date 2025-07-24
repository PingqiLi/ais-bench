from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import ShareGPTDataset, ShareGPTEvaluator, math_postprocess_v2


sharegpt_reader_cfg = dict(
    input_columns=['human'],
    output_column='gpt'
)


sharegpt_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={'type': 'conversations', 'prompt': "human"}
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

sharegpt_eval_cfg = dict(
    evaluator=dict(type=ShareGPTEvaluator)
)

sharegpt_datasets = [
    dict(
        abbr='sharegpt',
        type=ShareGPTDataset,
        disable_shuffle=True,
        path='ais_bench/datasets/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=sharegpt_reader_cfg,
        infer_cfg=sharegpt_infer_cfg,
        eval_cfg=sharegpt_eval_cfg
    )
]