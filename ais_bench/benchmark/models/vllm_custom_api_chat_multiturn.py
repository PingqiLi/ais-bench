import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Union, Tuple
from mmengine.config import ConfigDict

from tqdm import tqdm

from openai import OpenAI

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base_api import BaseAPIModel, handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel
from ais_bench.benchmark.utils.build import build_client_from_cfg

PromptType = Union[PromptList, str, dict]


@MODELS.register_module()
class VllmMultiturnAPIChatStream(PerformanceAPIModel):
    pass