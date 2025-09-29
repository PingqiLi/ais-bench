"""Direct Generation Inferencer."""

import asyncio
from multiprocessing import BoundedSemaphore
from typing import List, Optional

import aiohttp
import torch
from tqdm import tqdm

from ais_bench.benchmark.models.output import RequestOutput
from ais_bench.benchmark.registry import ICL_INFERENCERS
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.utils.logging import get_logger
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_api_inferencer import BaseApiInferencer
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_local_inferencer import BaseLocalInferencer
from ais_bench.benchmark.openicl.icl_inferencer.output_handler.gen_inferencer_output_handler import GenInferencerOutputHandler


logger = get_logger(__name__)


@ICL_INFERENCERS.register_module()
class GenInferencer(BaseApiInferencer, BaseLocalInferencer):
    """Generation Inferencer class to directly evaluate by generation.

    Attributes:
        model_cfg
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters. Defaults to 1.
    """

    def __init__(
        self,
        model_cfg,
        stopping_criteria: List[str] = [],
        batch_size: Optional[int] = 1,
        mode: Optional[str] = "infer",
        gen_field_replace_token: Optional[str] = "",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            batch_size=batch_size,
            mode=mode,
            output_json_filepath=output_json_filepath,
            save_every=save_every,
            **kwargs,
        )

        self.stopping_criteria = list(stopping_criteria) if stopping_criteria else []
        self.gen_field_replace_token = gen_field_replace_token or ""

        self.output_handler = GenInferencerOutputHandler(
            model=self.model, perf_mode=self.perf_mode
        )

    async def do_request(
        self, data: dict, token_bucket: BoundedSemaphore, session: aiohttp.ClientSession
    ):
        """Execute a single inference request.
        
        Args:
            data: Dictionary containing request data
            token_bucket: Semaphore for rate limiting
            session: HTTP session for the request
        """
        index = data.pop("index")
        input = data.pop("prompt")
        data_abbr = data.pop("data_abbr")
        max_out_len = data.pop("max_out_len")
        gold = data.pop("gold", None)
        output = RequestOutput(self.perf_mode)
        await self.status_counter.post()
        await self.model.generate(input, max_out_len, output, session=session, **data)
        if output.success:
            await self.status_counter.rev()
        else:
            await self.status_counter.failed()
        await self.status_counter.finish()
        await asyncio.to_thread(
            self.output_handler.report_cache_info, index, input, output, data_abbr, gold
        )

    def batch_inference(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> List:
        """Perform batch inference on the given dataloader.
        
        Args:
            dataloader: DataLoader containing the inference data
            
        Returns:
            List of inference results
        """
        logger.info("Starting inference process...")
        for datum in tqdm(dataloader, disable=not self.is_main_process):
            indexs = datum.pop("index")
            inputs = datum.pop("prompt")
            data_abbrs = datum.pop("data_abbr")
            max_out_lens = datum.pop("max_out_len")
            golds = datum.pop("gold", [None] * len(inputs))
            outputs = self.model.generate(inputs, max_out_lens, **datum)
            # TODO: save output to json
            for index, input, output, data_abbr, gold in zip(
                indexs, inputs, outputs, data_abbrs, golds
            ):
                self.output_handler.report_cache_info(
                    index, input, output, data_abbr, gold
                )

    def get_data_list(
        self,
        retriever: BaseRetriever,
    ):
        """Generate data list for inference.
        
        Args:
            retriever: The retriever instance to get data from
            
        Returns:
            List of data dictionaries for inference
        """
        # TODO: reuse mode, load tmp results only and infer unprocessed data
        data_abbr = retriever.dataset.abbr
        ice_idx_list = retriever.retrieve()
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=self.gen_field_replace_token,
            )
            parsed_prompt = self.model.parse_template(prompt, mode="gen")
            prompt_list.append(parsed_prompt)
        gold_ans = retriever.get_gold_ans()
        data_list = []
        for index, prompt in enumerate(prompt_list):
            # Required field information
            data_list.append(
                {
                    "prompt": prompt,
                    "data_abbr": data_abbr,
                    "index": index,
                    "max_out_len": self.model.max_out_len,
                }
            )
        if gold_ans is not None:
            for index, gold in enumerate(gold_ans):
                data_list[index]["gold"] = gold

        # Dataset-specified max_out_len has highest priority
        max_out_lens = retriever.dataset_reader.get_max_out_len()
        if max_out_lens is not None:
            for index, max_out_len in enumerate(max_out_lens):
                data_list[index]["max_out_len"] = (
                    max_out_len if max_out_len else self.model.max_out_len
                )

        return data_list