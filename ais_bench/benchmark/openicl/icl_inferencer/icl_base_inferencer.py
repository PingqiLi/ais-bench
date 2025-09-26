"""Basic Inferencer."""

import os
import os.path as osp
from abc import abstractmethod
from typing import List, Optional

from mmengine.dist import is_main_process


from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import (
    BaseInferencerOutputHandler,
)
from ais_bench.benchmark.openicl.icl_prompt_template import PromptTemplate
from ais_bench.benchmark.openicl.icl_retriever import BaseRetriever
from ais_bench.benchmark.utils import (
    get_logger,
    build_model_from_cfg,
    model_abbr_from_cfg,
)


MAX_BATCH_SIZE = 100000
logger = get_logger(__name__)


class BaseInferencer:
    """Base Inferencer class for all evaluation Inferencer.

    Attributes:
        model_cfg (Config): model config.
        model: built model instance (returned by build_model_from_cfg).
        batch_size (int): batch size for processing.
        output_json_filepath (str): output json path or directory.
        output_handler: output handler instance (provided by BaseInferencerOutputHandler).
        is_main_process (bool): whether the current process is the main process (distributed scenario).
    """

    def __init__(
        self,
        model_cfg,
        batch_size: Optional[int] = 1,
        output_json_filepath: Optional[str] = "./icl_inference_output",
    ) -> None:
        # basic parameters normalization
        self.model_cfg = model_cfg
        self.batch_size = int(batch_size) if batch_size else 1

        if self.batch_size < 1 or self.batch_size > MAX_BATCH_SIZE:
            raise ValueError(
                f"The range of batch_size is [1, {MAX_BATCH_SIZE}], but got {self.batch_size}. "
                "Please set it in datasets config"
            )

        # save output path (subclass does not need to repeat assignment)
        self.output_json_filepath = output_json_filepath

        # construct model and output handler (if needed, can be changed to lazy build)
        self.model = build_model_from_cfg(model_cfg)
        self.output_handler = BaseInferencerOutputHandler(self.model)

        # identify whether the current process is the main process (avoid covering the method with boolean)
        self.is_main_process = self._is_main_process()

    @abstractmethod
    def get_data_list(
        self,
        retriever: BaseRetriever,
        ice_template: Optional[PromptTemplate] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ) -> List:
        """Get the data list for inference."""

        raise NotImplementedError(f"{self.__class__.__name__} should be implemented")

    def _is_main_process(self):
        if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
            return int(os.getenv("RANK", "0")) == 0
        return is_main_process()

    def get_output_dir(self, output_json_filepath: Optional[str] = None):
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        output_json_filepath = osp.join(
            output_json_filepath,
            "performances" if self.perf_mode else "predictions",
            model_abbr_from_cfg(self.model_cfg),
        )
        return output_json_filepath
