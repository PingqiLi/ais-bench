import json
import os
import random
import re
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import httpx
import jieba
import requests
from tqdm import tqdm

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base_api import BaseAPIModel, handle_synthetic_input
from ais_bench.benchmark.models.base_api import BaseAPIModel, handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel
from ais_bench.benchmark.clients import TritonStreamClient

PromptType = Union[PromptList, str]


@MODELS.register_module()
class TritonCustomAPI(BaseAPIModel):
    """Model wrapper around Triton's models. TGI 0.9.4

    Args:
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
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
                 model_name: str = "",
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 generation_kwargs: Optional[Dict] = None):
        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.model_name = model_name
        self.base_url = self._get_base_url()
        super().__init__(path="",
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         verbose=verbose,
                         generation_kwargs=generation_kwargs)
        self.logger.info("Running triton model name is: " + self.model_name)

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
        """Generate results given a list of inputs.

        Args:
            inputs (PromptType): A string or PromptDict.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            str: The generated string.
        """
        assert isinstance(input, str)

        if max_out_len <= 0:
            return ''

        max_num_retries = 0
        while max_num_retries < self.retry:
            max_num_retries += 1
            header = {
                'Content-Type': 'application/json',
            }

            try:
                parameters_dict = self.generation_kwargs
                parameters_dict["max_new_tokens"] = max_out_len
                data = dict(
                    id=str(uuid.uuid4()),
                    text_input=input,
                    parameters=parameters_dict,
                )
                url = os.path.join(self.base_url, f"v2/models/{self.model_name}/generate")
                raw_response = requests.post(url, headers=header, data=json.dumps(data))

            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                self.wait()
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                continue
            self.logger.debug(str(response))
            if response.get('text_output') is None:
                raise ValueError(f"Unexpect response: {response}")
            return response['text_output']

        raise RuntimeError('Calling triton text API failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/"
        return f"http://{self.host_ip}:{self.host_port}/"

@MODELS.register_module()
class TritonCustomAPIStream(PerformanceAPIModel):
    """Model wrapper around Triton's models. TGI 0.9.4

    Args:
        max_seq_len (int): The maximum allowed sequence length of a model.
            Note that the length of prompt + generated tokens shall not exceed
            this value. Defaults to 2048.
        query_per_second (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
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
                 model_name: str = "",
                 path: str = "",
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 generation_kwargs: Optional[Dict] = None):
        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.model_name = model_name
        self.base_url = self._get_base_url()
        self.generation_kwargs = generation_kwargs
        self.endpoint_url = os.path.join(self.base_url, f"v2/models/{self.model_name}/generate_stream")
        self.client = TritonStreamClient(self.endpoint_url)

        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         verbose=verbose,
                         generation_kwargs=generation_kwargs)
        self.logger.info("Running triton model name is: " + self.model_name)

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
        assert isinstance(input, str)
        if max_out_len <= 0:
            return ''
        cache_data = self.prepare_input_data(input)
        self.generation_kwargs.update({"max_new_tokens": max_out_len})

        max_num_retries = 0
        while max_num_retries < self.retry:
            max_num_retries += 1
            try:
                response = self.client.request(cache_data, self.generation_kwargs)
                self.update_decode(cache_data)
            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                self.wait()
                continue
            except Exception as e:
                raise RuntimeError(f"Process response failed and the reason is {e}")

            self.logger.debug(str(response))
            return ''.join(response)

        raise RuntimeError('Calling Triton Stream API failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/"
        return f"http://{self.host_ip}:{self.host_port}/"