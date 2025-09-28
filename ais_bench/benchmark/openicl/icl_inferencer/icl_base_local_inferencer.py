"""Basic Inferencer."""
import os
import threading
import time
import uuid
from typing import List, Optional

import torch
from torch.utils.data import DataLoader

from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.openicl.utils import get_logger
from ais_bench.benchmark.openicl.icl_inferencer.icl_base_inferencer import BaseInferencer


MAX_BATCH_SIZE = 100000
logger = get_logger(__name__)


class BaseLocalInferencer(BaseInferencer):
    """Base Inferencer class for all evaluation Inferencer.

    Attributes:
        model_cfg (Config): model config.
        max_model_token_num (:obj:`int`, optional): Maximum number of
            tokenized words allowed by the LM.
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
    """

    def batch_inference(self, dataloader: torch.utils.data.DataLoader) -> List:
        raise NotImplementedError("Method hasn't been implemented yet")

    def inference(
        self,
        retriever: BaseRetriever,
        output_json_filepath: Optional[str] = None,
    ) -> List:
        # General inference interface that receives retriever, constructs dataloader,
        # and performs inference according to batch size. Mainly used for pure model inference.
        """Perform In-Context Inference given a retriever and optional
        templates.

        Args:
            retriever (:obj:`BaseRetriever`): An instance of a Retriever class
                that will be used to retrieve in-context examples
            output_json_filepath (:obj:`str`, optional): The file path to save
                the results as a `JSON` file. Defaults to None.

        Raises:
            NotImplementedError: If the function is not implemented in the
                subclass.
        """
        # Output path format: output_json_filepath/performances|predictions/model_abbr/dataset_abbr.jsonl
        # Cache data path: output_json_filepath/performances|predictions/model_abbr/tmp/tmp_{timestamp}_{uuid}.jsonl
        # tmp_{timestamp}_{uuid}.json stores data from output_handler.results_dict, cached in single-line format: {data_abbr: {index: {}}}
        out_path = self.get_output_dir(output_json_filepath)
        # Save temporary results
        tmp_json_filepath = os.path.join(out_path, "tmp")
        os.makedirs(tmp_json_filepath, exist_ok=True)
        tmp_file_name = (
            f"tmp_{time.strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().split('-')[0]}.jsonl"
        )
        cache_consumer_thread = threading.Thread(
            target=self.output_handler.run_cache_consumer,
            args=(tmp_json_filepath, tmp_file_name, self.perf_mode, self.save_every),
        )
        cache_consumer_thread.start()

        # Generate prompts for testing input
        if isinstance(retriever, List) and all(
            isinstance(r, BaseRetriever) for r in retriever
        ):
            data_list = []
            for r in retriever:
                data_list.extend(self.get_data_list(r, self.gen_field_replace_token))
        elif isinstance(retriever, BaseRetriever):
            data_list = self.get_data_list(retriever, self.gen_field_replace_token)
        else:
            raise ValueError(
                "retriever must be a BaseRetriever or a List of BaseRetriever"
            )
        logger.info("Starting to build dataloader")
        dataloader = self.get_dataloader(data_list, self.batch_size)
        try:
            # Execute inference tasks according to batch size
            self.batch_inference(dataloader)
        finally:
            self.output_handler.stop_cache_consumer()
            cache_consumer_thread.join()

        logger.info("Inference process finished")
        # Handle cache data
        if self.is_main_process:
            os.makedirs(out_path, exist_ok=True)
            self.output_handler.write_to_json(out_path)
            self.output_handler.clean_cache_dir(tmp_json_filepath)

    @staticmethod
    def get_dataloader(datalist: List[List], batch_size: int) -> DataLoader:
        """Return a dataloader of the input data list."""

        def custom_collate_fn(batch):
            return {key: [d[key] for d in batch] for key in batch[0]}

        dataloader = DataLoader(
            datalist, batch_size=batch_size, collate_fn=custom_collate_fn
        )
        return dataloader