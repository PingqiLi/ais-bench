"""
Generation inferencer output handler.

This module provides a specialized output handler for generation-based
inference tasks, supporting both performance and accuracy modes.
"""

import queue
import traceback
from typing import Any, Dict, List, Optional, Union

import h5py
from .base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils import get_logger

logger = get_logger(__name__)


class GenInferencerOutputHandler(BaseInferencerOutputHandler):
    """
    Output handler for generation-based inference tasks.

    This handler specializes in processing generation model outputs,
    supporting both performance measurement and accuracy evaluation modes.
    It handles different data formats and provides appropriate result storage.

    Attributes:
        all_success (bool): Flag indicating if all operations were successful
        perf_mode (bool): Whether in performance measurement mode
        cache_queue (queue.Queue): Queue for caching results before writing
    """

    def __init__(self, model: Any, perf_mode: bool = False) -> None:
        """
        Initialize the generation inferencer output handler.

        Args:
            model: The model instance for inference operations
            perf_mode (bool): Whether to run in performance measurement mode
                            (default: False for accuracy mode)

        Raises:
            TypeError: If model is None or invalid
        """
        super().__init__(model)
        self.all_success = True
        self.perf_mode = perf_mode

    def load_tmp_result(self, tmp_path: str, file_format: str = "jsonl") -> None:
        """
        Load temporary results from file.

        This method is currently not implemented and raises NotImplementedError.
        Future implementation should handle loading of temporary result files.

        Args:
            tmp_path (str): Path to the temporary result file
            file_format (str): Format of the file (default: "jsonl")

        Raises:
            NotImplementedError: This method is not yet implemented
        """
        raise NotImplementedError("load_tmp_result is not implemented")

    def get_result(
        self,
        h5_group: h5py.Group,
        input: Union[str, List[str]],
        output: Union[str, Output],
        gold: Optional[str] = None,
    ) -> dict:
        """
        Save inference results to the results dictionary.

        Handles both performance and accuracy modes with different data storage
        strategies. In performance mode, only metrics are stored. In accuracy mode,
        full input/output data is preserved for evaluation.

        Args:
            h5_group (h5py.Group): HDF5 group to write results to
            input (Union[str, List[str]]): Input data for the inference
            output (Union[str, Output]): Output result from inference
            gold (Optional[str]): Ground truth data for comparison

        Raises:
            KeyError: If output object is invalid
            ValueError: If input parameters are invalid
        """
        try:

            # Performance mode: only store metrics
            if self.perf_mode and isinstance(output, Output):
                try:
                    result_data = output.get_metrics()
                    result_data = self._extract_and_write_arrays(
                        result_data, h5_group
                    )
                except Exception as e:
                    logger.error(f"Failed to get metrics from output: {str(e)}")
                    logger.error(f"Exception details: {traceback.format_exc()}")
                    raise
            else:
                # Accuracy mode: store full input/output data
                try:
                    result_data = {
                        "success": (
                            output.success if isinstance(output, Output) else True
                        ),
                        "origin_prompt": input,
                        "prediction": (
                            output.get_prediction()
                            if isinstance(output, Output)
                            else output
                        ),
                    }

                    if gold:
                        result_data["gold"] = gold

                except Exception as e:
                    logger.error(f"Failed to process output data: {str(e)}")
                    logger.error(f"Exception details: {traceback.format_exc()}")
                    raise

            # Check for failures and update success status
            if not result_data.get("success", True):
                self.all_success = False
                if isinstance(output, Output) and hasattr(output, "error_info"):
                    result_data["error_info"] = output.error_info
                else:
                    logger.warning(
                        f"No error info available for failed operation at data id {id}"
                    )
            return result_data

        except Exception as e:
            logger.error(f"Error in save_results for data id {id}: {str(e)}")
            logger.error(f"Exception details: {traceback.format_exc()}")
            raise
