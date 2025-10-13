import time
from abc import abstractmethod

import numpy as np
from ais_bench.benchmark.utils import get_logger


class Output:
    def __init__(self, perf_mode: bool = False) -> None:
        self.perf_mode = perf_mode
        self.input: list | str = None
        self.content: list[str] | str = ""
        self.reasoning_content: list[str] | str = ""
        self.success: bool = False
        self.time_points: list[float] = []
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.latency: float = 0.0
        self.error_info: str = ""
        self.extra_perf_data: dict = {}
        self.extra_details_data: dict = {}
        self.logger = get_logger()
        self.is_mm_prompt: bool = False

    def _get_input_tokens(self, model):
        """Calculate the actual input token length for the model.
        
        Args:
            model: The model instance with encode method
        """
        if self.input_tokens:
            return
        if self.is_mm_prompt:
            return
        if hasattr(model, "encode"):
            token_ids = model.encode(self.input)
            self.input_tokens = len(token_ids)

    def _get_output_tokens(self, model):
        """Calculate the actual output token length for the model.
        
        Args:
            model: The model instance with encode method
        """
        if self.output_tokens:
            return
        if hasattr(model, "encode"):
            token_ids = model.encode(self.content) + model.encode(
                self.reasoning_content
            )
            self.output_tokens = len(token_ids)

    def get_metrics(self, model) -> dict:
        """Calculate and return performance metrics for the output.
        
        Args:
            model: The model instance used for token calculation
            
        Returns:
            dict: Cleaned metrics dictionary with performance data
        """
        def clean_result(res):
            for key in ["input", "content", "reasoning_content", "perf_mode"]:
                res.pop(key, None)
            return res

        if not self.success:
            return clean_result(self.to_dict())
        self._get_input_tokens(model)
        self._get_output_tokens(model)
        self.time_points = np.array(self.time_points, dtype=np.float64)
        if self.time_points.size < 2:
            self.success = False
            self.error_info = "chunk size is less than 2"
        self.latency = self.time_points[-1] - self.time_points[0] if self.success else 0
        return clean_result(self.to_dict())

    def _concate_reasoning_content(self, content, reasoning_content) -> str:
        """Concatenate reasoning content with main content.
        
        Args:
            content: Main content string
            reasoning_content: Reasoning content string
            
        Returns:
            str: Combined content with reasoning
        """
        if reasoning_content:
            if content:
                return reasoning_content + "</think>" + content
            else:
                return reasoning_content
        else:
            return content

    def get_prediction(self) -> dict:
        """Get the final prediction by combining content and reasoning.
        
        Returns:
            dict: Combined prediction content
        """
        if not self.reasoning_content:
            return self.content

        if isinstance(self.content, list) and isinstance(self.reasoning_content, list):
            return [
                self._concate_reasoning_content(content, reasoning_content)
                for content, reasoning_content in zip(
                    self.content, self.reasoning_content
                )
            ]
        elif isinstance(self.reasoning_content, str):
            return self._concate_reasoning_content(self.content, self.reasoning_content)

        return self.content

    def to_dict(self):
        """Convert all instance attributes to dictionary.
        
        Returns:
            dict: Dictionary containing all instance attributes
        """
        return self.__dict__

    async def record_time_point(self) -> None:
        """Record a time point for performance measurement.
        
        This method is called by the model to record timing data.
        """
        if self.perf_mode:
            self.time_points.append(time.perf_counter())


class RequestOutput(Output):

    def __init__(self, perf_mode: bool = False) -> None:
        super().__init__(perf_mode)
        self.ttft: float = 0.0
        self.tpot: float = 0.0
        self.itl: list[float] = []
        self.start_time: float = 0.0
        self.end_time: float = 0.0
        self.generate_tokens_speed: float = 0.0

    def get_metrics(self, model) -> dict:
        """Calculate and return detailed performance metrics for request output.
        
        Args:
            model: The model instance used for token calculation
            
        Returns:
            dict: Enhanced metrics dictionary with request-specific performance data
        """
        result = super().get_metrics(model)
        if not self.success:
            result.update(self.to_dict())
            # Keep only ITL, time_points is not needed
            result.pop("time_points")
            return result
        self.ttft = float(self.time_points[1] - self.time_points[0])
        self.itl = np.diff(self.time_points)[1:] if self.time_points.size > 2 else []
        self.tpot = (
            (self.latency - self.ttft) / (self.output_tokens - 1)
            if self.output_tokens > 1
            else 0
        )
        self.start_time = float(self.time_points[0])
        self.end_time = float(self.time_points[-1])
        self.generate_tokens_speed = self.output_tokens / self.latency
        result.update(self.to_dict())
        # Keep only ITL, time_points is not needed
        result.pop("time_points")
        return result
