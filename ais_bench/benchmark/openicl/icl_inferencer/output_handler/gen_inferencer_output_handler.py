from typing import List, Optional, Union

import sqlite3
import uuid

from ais_bench.benchmark.openicl.icl_inferencer.output_handler.base_handler import BaseInferencerOutputHandler
from ais_bench.benchmark.models.output import Output
from ais_bench.benchmark.utils.logging.error_codes import ICLI_CODES
from ais_bench.benchmark.utils.logging.exceptions import AISBenchImplementationError

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

    def __init__(self, perf_mode: bool = False, save_every: int = 100) -> None:
        """
        Initialize the generation inferencer output handler.

        Args:
            perf_mode (bool): Whether to run in performance measurement mode
                            (default: False for accuracy mode)
        """
        super().__init__(save_every)
        self.all_success = True
        self.perf_mode = perf_mode

    def get_prediction_result(self, input: Union[str, List[str]], output: Union[str, Output], gold: Optional[str] = None) -> dict:
        result_data = {
            "success": (
                output.success if isinstance(output, Output) else True
            ),
            "uuid": output.uuid if isinstance(output, Output) else uuid.uuid4().hex[:8],
            "origin_prompt": input,
            "prediction": (
                output.get_prediction()
                if isinstance(output, Output)
                else output
            ),
        }
        if gold:
            result_data["gold"] = gold
        return result_data