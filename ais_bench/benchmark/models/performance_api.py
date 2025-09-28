import os
import time
import uuid
import multiprocessing
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
import copy
from mmengine.config import ConfigDict

from ais_bench.benchmark.utils.tokenizer import BenchmarkTokenizer
from ais_bench.benchmark.models.base_api import BaseAPIModel


class PerformanceAPIModel(BaseAPIModel):
    pass