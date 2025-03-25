from mmengine.config import read_base

with read_base():
    # gsm8k
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_0_shot_cot_chat
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_0_shot_cot_str import gsm8k_datasets as gsm8k_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_chat_prompt import gsm8k_datasets as gsm8k_4_shot_cot_chat
    from ais_bench.benchmark.configs.datasets.gsm8k.gsm8k_gen_4_shot_cot_str import gsm8k_datasets as gsm8k_4_shot_cot_str

    # ceval
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_cot_chat_prompt import ceval_datasets as ceval_0_shot_chat
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_0_shot_str import ceval_datasets as ceval_0_shot_str
    from ais_bench.benchmark.configs.datasets.ceval.ceval_gen_5_shot_str import ceval_datasets as ceval_5_shot_str

    # drop
    from ais_bench.benchmark.configs.datasets.drop.drop_gen_0_shot_str import drop_datasets as drop_0_shot_str
    from ais_bench.benchmark.configs.datasets.drop.drop_gen_3_shot_str import drop_datasets as drop_3_shot_str

    # gpqa
    from ais_bench.benchmark.configs.datasets.gpqa.gpqa_gen_0_shot_str import gpqa_datasets as gpqa_0_shot_str

    # aime2024
    from ais_bench.benchmark.configs.datasets.aime2024.aime2024_gen_0_shot_str import aime2024_datasets as aime2024_0_shot_str

    # humaneval
    from ais_bench.benchmark.configs.datasets.humaneval.humaneval_gen_0_shot import humaneval_datasets as humaneval_0_shot_str

    # math
    from ais_bench.benchmark.configs.datasets.math.math_prm800k_500_0shot_cot_gen import math_datasets as math500_0_shot_str
    from ais_bench.benchmark.configs.datasets.math.math_prm800k_500_5shot_cot_gen import math_datasets as math500_5_shot_str

    # mmlu
    from ais_bench.benchmark.configs.datasets.mmlu.mmlu_gen_5_shot_str import mmlu_datasets as mmlu_5_shot_str

    # mmlu_pro
    from ais_bench.benchmark.configs.datasets.mmlu_pro.mmlu_pro_gen_0_shot_str import mmlu_pro_datasets as mmlu_pro_0_shot_str
    from ais_bench.benchmark.configs.datasets.mmlu_pro.mmlu_pro_gen_5_shot_str import mmlu_pro_datasets as mmlu_pro_5_shot_str

    # boolq
    from ais_bench.benchmark.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_0_shot_cot_str import BoolQ_datasets as boolq_0_shot_cot_str
    from ais_bench.benchmark.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_0_shot_str import BoolQ_datasets as boolq_0_shot_str
    from ais_bench.benchmark.configs.datasets.SuperGLUE_BoolQ.SuperGLUE_BoolQ_gen_5_shot_str import BoolQ_datasets as boolq_5_shot_str
