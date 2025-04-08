import os
import sys
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import transformers

from ais_bench.benchmark.models.base import BaseModel
from ais_bench.benchmark.models.base_api import APITemplateParser
from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.logging import get_logger
from ais_bench.benchmark.utils.prompt import PromptList


DTYPE_MAP = {"bf16": torch.bfloat16, "fp16": torch.float16}


@MODELS.register_module()
class MindieLLMAPI(BaseModel):
    """
    Model wrapper around MindIE-LLM models.
    """

    def __init__(self,
                 environ_kwargs: Optional[Dict] = None,
                 **kwargs):

        self.rank = int(os.getenv("RANK", "0"))
        self.local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.world_size = kwargs.get('world_size')
        self.block_size = kwargs.get('block_size')

        self.model_name = kwargs.get('model_name')
        self.data_type = kwargs.get('data_type')  # fp16 / bf16
        self.weight_dir = kwargs.get('weight_dir')
        self.max_position_embedding = kwargs.get('max_position_embedding')
        self.is_chat_model = kwargs.get('is_chat_model')
        self.prefill_batch_size = kwargs.get('prefill_batch_size')
        self.kw_args = kwargs.get('kw_args')
        self.dp = kwargs.get('dp')
        self.tp = kwargs.get('tp')
        self.sp = kwargs.get('sp')
        self.moe_tp = kwargs.get('moe_tp')
        self.pp = kwargs.get('pp')
        self.microbatch_size = kwargs.get('microbatch_size')
        self.moe_ep = kwargs.get('moe_ep')
        self.trust_remote_code = kwargs.get('trust_remote_code')
        self.ignore_eos = kwargs.get('ignore_eos')
        self.input_length = kwargs.get('input_length')
        self.output_length = kwargs.get('output_length')
        self.decode_batch_size = kwargs.get('decode_batch_size')
        self.logger = get_logger()
        self.pa_runner = None

        self.get_model_or_runner(self.input_length, self.output_length)
        self.check_pa_runner()
        self.warm_up()

        super().__init__(path=self.weight_dir,
                         max_seq_len=self.output_length,
                         tokenizer_only=False,
                         meta_template=None)
        
        for key, value in environ_kwargs.items():
            os.environ[key] = value

    def check_pa_runner(self):
        if self.pa_runner == None:
            raise RuntimeError("Model loading failed")
        

    def warm_up(self):
        self.pa_runner.warm_up()


    def get_model_or_runner(self, input_length, output_length, warmup_bs=0):

        try:
            ATB_SPEED_HOME_PATH = os.environ.get("ATB_SPEED_HOME_PATH")
            if ATB_SPEED_HOME_PATH not in sys.path:
                sys.path.insert(0, os.path.join(ATB_SPEED_HOME_PATH, "../.."))
                sys.path.insert(0, ATB_SPEED_HOME_PATH)
            from atb_llm.utils.env import ENV
            from examples.run_pa import PARunner
        except Exception:
            raise RuntimeError("Failed to import necessary packages")

        rank = "rank"
        world_size = "world_size"
        local_rank = "local_rank"
        model_path = "model_path"
        max_position_embeddings = "max_position_embeddings"
        max_input_length = "max_input_length"
        max_output_length = "max_output_length"
        trust_remote_code = "trust_remote_code"


        prefill_batch_size = self.decode_batch_size if self.prefill_batch_size == 0 else self.prefill_batch_size

        input_dict = {
            rank: self.rank,
            local_rank: self.local_rank,
            world_size: self.world_size,
            'max_prefill_tokens': -1,
            'block_size': self.block_size,
            model_path: self.weight_dir,
            max_position_embeddings: (self.max_position_embedding
                                        if self.max_position_embedding != -1
                                        else input_length + output_length),
            'max_prefill_batch_size': prefill_batch_size,
            'max_batch_size': warmup_bs if warmup_bs != 0 else self.decode_batch_size,
            max_input_length: input_length,
            max_output_length: output_length,
            'kw_args': self.kw_args,
            'dp': self.dp,
            'tp': self.tp,
            'sp': self.sp,
            'moe_tp': self.moe_tp,
            'pp': self.pp,
            'microbatch_size': self.microbatch_size,
            'moe_ep': self.moe_ep,
            trust_remote_code: self.trust_remote_code
        }
        if self.model_name == "qwen2_72b" or self.model_name == "qwen2_7b":
            input_dict[max_position_embeddings] = None
        self.pa_runner = PARunner(**input_dict)
        model_dtype = self.pa_runner.model.dtype
        self.tokenizer = self.pa_runner.model.tokenizer
        user_dtype = DTYPE_MAP.get(self.data_type, None)
        if user_dtype != model_dtype:
            self.logger.error(
                "Inconsistent dtype: Input dtype: %s, model weight dtype: %s. please check",
                user_dtype, model_dtype)
            raise RuntimeError(
                f"Inconsistent dtype: Input dtype: {user_dtype}, " +
                f"model weight dtype: {model_dtype}. please check")

        self.logger.info('%d pa_runner: %s', self.rank, self.pa_runner)


    def generate(self,
                 inputs: List[str],
                 max_out_len: int,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        with torch.no_grad():
            generate_texts, _, _ = self.pa_runner.infer(inputs,
                                                        len(inputs),
                                                        max_out_len,
                                                        self.ignore_eos,
                                                        self.is_chat_model)
            
        return generate_texts


    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))

