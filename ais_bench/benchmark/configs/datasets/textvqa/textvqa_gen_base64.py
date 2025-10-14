from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import TEXTVQADataset, TEXTEvaluator, math_postprocess_v2


textvqa_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)


textvqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={'type': "image_text", 'data': ['image_url_base64', 'text'], 'prompt': " Answer the question using a single word or phrase."}
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

textvqa_eval_cfg = dict(
    evaluator=dict(type=TEXTEvaluator)
)

textvqa_datasets = [
    dict(
        abbr='textvqa',
        type=TEXTVQADataset,
        path='ais_bench/datasets/textvqa/textvqa_json/textvqa_val.jsonl', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=textvqa_reader_cfg,
        infer_cfg=textvqa_infer_cfg,
        eval_cfg=textvqa_eval_cfg
    )
]