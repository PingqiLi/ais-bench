from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union
from mmengine.config import ConfigDict

import requests
from tqdm import tqdm
from torch.utils.data import DataLoader

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.clients import MindieStreamClient
from ais_bench.benchmark.models.base_api import handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel

PromptType = Union[PromptList, str, dict]


@MODELS.register_module()
class MindieStreamApi(PerformanceAPIModel):
    """Model wrapper around OpenAI's models.

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

    is_api: bool = True

    def __init__(self,
                 path,
                 max_seq_len: int = 4096,
                 request_rate: int = 1,
                 traffic_cfg: Optional[ConfigDict] = None,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 custom_client = MindieStreamClient, # BaseClient
                 generation_kwargs: Optional[Dict] = None,
                 trust_remote_code: bool = False,
                 ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         request_rate=request_rate,
                         traffic_cfg=traffic_cfg,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         generation_kwargs=generation_kwargs,
                         verbose=verbose,
                         trust_remote_code=trust_remote_code
                         )
        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.url = self._get_base_url()
        self.client = custom_client(self.url, retry)

    def generate(self,
                 inputs: List[PromptType],
                 max_out_len: int = 512,
                 **kwargs) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        batch_size = kwargs.get("batch_size", len(inputs))
        with ThreadPoolExecutor(max_workers=batch_size) as executor:
            results = list(
                tqdm(executor.map(self._generate, inputs,
                                  [max_out_len] * len(inputs)),
                     total=len(inputs),
                     desc='Inferencing'))
        return results

    @handle_synthetic_input
    def _generate(self, input: PromptType, max_out_len: int) -> str:
        """Generate result given a input.

        Args:
            input (PromptType): A string or PromptDict.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        if isinstance(input, dict):
            data_id = input.get('data_id')
            input = input.get('prompt')
        else:
            data_id = -1

        if max_out_len <= 0:
            return ''
        cache_data = self.prepare_input_data(input, data_id)
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_new_tokens": max_out_len})

        response = self.client.request(cache_data, generation_kwargs)
        self.set_result(cache_data)
        return ''.join(response)

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/infer"
        return f"http://{self.host_ip}:{self.host_port}/infer"