from mmengine.config import read_base

with read_base():
    from .gsm8k_gen_ee684f import gsm8k_datasets  # noqa: F401, F403
