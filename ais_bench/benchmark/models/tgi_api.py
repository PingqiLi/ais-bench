import os
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from mmengine.config import ConfigDict

from tqdm import tqdm

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base_api import handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel
from ais_bench.benchmark.utils.build import build_client_from_cfg

PromptType = Union[PromptList, str]


@MODELS.register_module()
class TGICustomAPI(PerformanceAPIModel):
    """Model wrapper around TGI's models. TGI 0.9.4

    Args:
        path (str): The path of model.
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        request_rate (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        traffic_cfg (ConfigDict, optional): control the request traffic rate 
            "burstiness": Optional[float],    # Burstiness factor controlling interval randomness (≥0, default:0)
            "ramp_up_strategy": Optional[str],  # Ramp-up strategy type ("linear", "exponential", or None)
            "ramp_up_start_rps": Optional[float],  # Starting RPS for ramp-up (required with strategy)
            "ramp_up_end_rps": Optional[float]   # Ending RPS for ramp-up (required with strategy)
        retry (int): Number of retires if the API call fails. Defaults to 2.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        host_ip (str): The  host ip of custom service, default "localhost".
        host_port (int): The host port of custom service, default "8080".
        enable_ssl (bool, optional): .
    """
    pass


@MODELS.register_module()
class TGICustomAPIStream(PerformanceAPIModel):
    """Model wrapper around TGI's models. TGI 0.9.4

    Args:
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        request_rate (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        traffic_cfg (ConfigDict, optional): control the request traffic rate 
            "burstiness": Optional[float],    # Burstiness factor controlling interval randomness (≥0, default:0)
            "ramp_up_strategy": Optional[str],  # Ramp-up strategy type ("linear", "exponential", or None)
            "ramp_up_start_rps": Optional[float],  # Starting RPS for ramp-up (required with strategy)
            "ramp_up_end_rps": Optional[float]   # Ending RPS for ramp-up (required with strategy)
        retry (int): Number of retires if the API call fails. Defaults to 2.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        host_ip (str): The  host ip of custom service, default "localhost".
        host_port (int): The host port of custom service, default "8080".
        enable_ssl (bool, optional): .
    """
    pass