import json
import os
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import httpx
import jieba
import requests
from tqdm import tqdm

from openai import OpenAI

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class VLLMCustomAPI(BaseAPIModel):
    """Model wrapper around OpenAI's models. vllm 0.6 +

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
        self.base_url = self._get_base_url()
        path = self._get_service_model_path()
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         verbose=verbose,
                         generation_kwargs=generation_kwargs)

        self.logger.info("Running model path name is: " + path)
        self.path = path

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
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(self._generate, inputs,
                                  [max_out_len] * len(inputs)),
                     total=len(inputs),
                     desc='Inferencing'))
        return results

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
            self.wait()
            max_num_retries += 1
            header = {
                'Content-Type': 'application/json',
            }

            try:
                data = dict(
                    model=self.path,
                    prompt=input,
                    max_tokens=max_out_len,
                )
                data = data | self.generation_kwargs
                url = os.path.join(self.base_url, "completions")
                raw_response = requests.post(url, headers=header, data=json.dumps(data))

            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                continue
            self.logger.debug(str(response))
            return response['choices'][0]['text']

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/v1"
        return f"http://{self.host_ip}:{self.host_port}/v1"

    def _get_service_model_path(self):
        client = OpenAI(api_key="EMPTY", base_url=self.base_url)
        return client.models.list().data[0].id


@MODELS.register_module()
class VLLMCustomAPIOld(BaseAPIModel):
    """Model wrapper around OpenAI's models. vllm 0.2.6

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
        self.base_url = self._get_base_url()
        path = "unused"
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         verbose=verbose,
                         generation_kwargs=generation_kwargs)

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
        with ThreadPoolExecutor() as executor:
            results = list(
                tqdm(executor.map(self._generate, inputs,
                                  [max_out_len] * len(inputs)),
                     total=len(inputs),
                     desc='Inferencing'))
        return results

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
            self.wait()
            max_num_retries += 1
            header = {
                'Content-Type': 'application/json',
            }

            try:
                data = dict(
                    prompt=input,
                    max_tokens=max_out_len,
                )
                data = data | self.generation_kwargs
                url = os.path.join(self.base_url, "generate")
                raw_response = requests.post(url, headers=header, data=json.dumps(data))

            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            try:
                response = raw_response.json()
            except requests.JSONDecodeError:
                self.logger.error('JsonDecode error, got',
                                  str(raw_response.content))
                continue

            if response.get('text') is None:
                raise RuntimeError("Get unexpected response: {response} from service!")
            return response['text'][0]

        raise RuntimeError('Calling VLLM failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}"
        return f"http://{self.host_ip}:{self.host_port}"
