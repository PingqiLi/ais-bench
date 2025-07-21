from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.datasets import VideoBenchDataset, VideoBenchEvaluator, math_postprocess_v2


videobench_reader_cfg = dict(
    input_columns=['question'],
    output_column='answer'
)


videobench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template={'type': "video_text", 'data': ['video_url_base64', 'text'], 'num_frames': '5',
                'prompt1': '\n Among the choice_length options choice_list above,'
                            ' the one closest to the correct answer is:',
                'prompt2': " Please respond with only the corresponding options and do not provide any explanations" 
                        + " or additional information. ASSISTANT:"}
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

videobench_eval_cfg = dict(
    evaluator=dict(type=VideoBenchEvaluator)
)

videobench_datasets = [
    dict(
        abbr='videobench',
        type=VideoBenchDataset,
        path='ais_bench/datasets/videobench', # 数据集路径，使用相对路径时相对于源码根路径，支持绝对路径
        reader_cfg=videobench_reader_cfg,
        infer_cfg=videobench_infer_cfg,
        eval_cfg=videobench_eval_cfg
    )
]