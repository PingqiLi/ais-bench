"""
Custom Sampled Dataset Config - Mixed evaluation dataset

This config loads a custom dataset created by sampling from multiple sources:
- AIME2024: Math competition problems
- MATH500: Grade school math problems
- CEval: Chinese evaluation benchmark
- MMLU: Massive multitask language understanding
- LiveCodeBench: Code generation tasks

The dataset is created using tools/create_sampled_dataset.py

Usage:
    # Create the dataset first
    python3 tools/create_sampled_dataset.py --aime-count 30 --math-count 40 ...

    # Then run evaluation
    ais_bench --models vllm_api_general_chat \
              --datasets custom_sampled_eval_gen_0_shot_cot_chat_prompt \
              --mode all
"""

from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_benchmark.openicl.icl_retriever import ZeroRetriever
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer
from ais_bench.benchmark.openicl.icl_evaluator import AccEvaluator
from ais_bench.benchmark.datasets import CustomDataset
from ais_bench.benchmark.utils import first_option_postprocess


# Inference configuration with Chain-of-Thought prompting
custom_sampled_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='请仔细阅读以下问题并给出正确答案。如果是选择题，请选出正确选项。回答之前先一步步思考。\n\n{question}'
                ),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

# Evaluation configuration
custom_sampled_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    # Use first_option_postprocess to extract answer from CoT reasoning
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD')
)

# Dataset configuration
custom_sampled_datasets = [
    dict(
        type=CustomDataset,
        abbr='custom_sampled_eval',
        # Default path - can be overridden
        path='ais_bench/datasets/custom_sampled_eval.jsonl',
        reader_cfg=dict(
            input_columns=['question'],
            output_column='answer',
            # Optional: limit to first N samples for quick testing
            # test_range='[0:50]'
        ),
        infer_cfg=custom_sampled_infer_cfg,
        eval_cfg=custom_sampled_eval_cfg,
    )
]
