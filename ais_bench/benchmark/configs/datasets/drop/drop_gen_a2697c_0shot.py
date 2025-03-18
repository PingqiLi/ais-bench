from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import DropOpenAIDataset, DropOpenAIEvaluator


drop_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='answers',
    train_split='validation',
    test_split='validation',
)


drop_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template='{prompt}'
        ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, stopping_criteria=['---', 'Passage', 'Question', 'You will be asked'], max_out_len=32768),
    )

drop_eval_cfg = dict(evaluator=dict(type=DropOpenAIEvaluator))

drop_datasets = [
    dict(
        abbr='drop',
        type=DropOpenAIDataset,
        path='ais_bench/datasets/drop_simple_eval/dev.jsonl',
        reader_cfg=drop_reader_cfg,
        infer_cfg=drop_infer_cfg,
        eval_cfg=drop_eval_cfg)
]
