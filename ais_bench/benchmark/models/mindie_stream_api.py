from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Union

import requests
from tqdm import tqdm

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList
from ais_bench.benchmark.clients import MindieStreamClient
from ais_bench.benchmark.models.base_api import handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel

PromptType = Union[PromptList, str]


@MODELS.register_module()
class MindieStreamApi(PerformanceAPIModel):
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
                 path,
                 max_seq_len: int = 4096,
                 query_per_second: int = 1,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 meta_template: Optional[Dict] = None,
                 verbose: bool = False,
                 host_ip: str = "localhost",
                 host_port: int = 8080,
                 enable_ssl: bool = False,
                 generation_kwargs: Optional[Dict] = None,
                 ):
        super().__init__(path=path,
                         max_seq_len=max_seq_len,
                         meta_template=meta_template,
                         query_per_second=query_per_second,
                         rpm_verbose=rpm_verbose,
                         retry=retry,
                         generation_kwargs=generation_kwargs,
                         verbose=verbose,
                         )
        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.url = self._get_base_url()
        self.client = MindieStreamClient(self.url)

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
        batch_size = len(inputs)
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

        raise RuntimeError('Calling Mindie Api failed after retrying for '
                           f'{max_num_retries} times. Check the logs for '
                           'details.')

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/infer"
        return f"http://{self.host_ip}:{self.host_port}/infer"