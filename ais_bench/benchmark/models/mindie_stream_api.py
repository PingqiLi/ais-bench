import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base_api import BaseAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class MindieStreamApi(BaseAPIModel):
    """Model wrapper around OpenAI's models.

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
        self.max_chunk_size = 32*2048
        self.res_key = 'generated_text'
        self.base_url = self._get_base_url()
        super().__init__(path='',
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         generation_kwargs=generation_kwargs,
                         verbose=verbose)

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

        self.generation_kwargs.update({"max_new_tokens": max_out_len})

        max_num_retries = 0
        while max_num_retries < self.retry:
            self.wait()
            max_num_retries += 1
            header = {
                'Content-Type': 'application/json',
            }

            try:
                data = dict(
                    inputs=input,
                    stream=True,
                    parameters=self.generation_kwargs
                )
                response = list()
                url = os.path.join(self.base_url, "infer")
                raw_response = requests.post(url, headers=header, data=json.dumps(data), stream=True)
                for res_ in self.process_response(raw_response):
                    if self.res_key not in res_:
                        continue
                    response.append(res_[self.res_key])

            except requests.ConnectionError:
                self.logger.error('Got connection error, retrying...')
                continue
            except Exception as e:
                raise RuntimeError(f"Process response failed and the reason is {e}")

            self.logger.debug(str(response))
            return ''.join(response)

        raise RuntimeError('Calling OpenAI failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def process_response(self, response):
        for byte_line in response.iter_content(self.max_chunk_size):
            if byte_line == b"\n":
                continue
            cur_line = self.preprocess_cur_line(byte_line.decode())
            try:
                for json_content in self._stream_data_split(cur_line):
                    yield json_content
            except Exception as error:
                raise RuntimeError(f"Request failed and the reason is {error}, response from server is: {cur_line}")

    def preprocess_cur_line(self, cur_line: str) -> str:
        if cur_line == "engine callback timeout.":
            self.logger.error(f"Engine time out. The tokens generation might be incomplete.")
        return cur_line

    def _stream_data_split(self, stream_data_line):
        stream_data_line = stream_data_line.lstrip("data:").rstrip("\n\0")
        # 使用正则替换调整数据格式，使其符合 JSON 语法
        stream_data_line = re.sub(r'\}\s*data:{', '} ,{', stream_data_line)
        stream_data_line = '[' + stream_data_line + ']'
        json_obj_arr = json.loads(stream_data_line)
        return json_obj_arr

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}"
        return f"http://{self.host_ip}:{self.host_port}"