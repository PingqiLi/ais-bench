import os
from typing import Dict, List, Optional, Union, Tuple
from transformers import AutoTokenizer

from openai import OpenAI

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base_api import BaseAPIModel
from ais_bench.benchmark.models.output import RequestOutput

PromptType = Union[PromptList, str]


@MODELS.register_module()
class VLLMCustomAPIChat(BaseAPIModel):
    """Model wrapper around OpenAI's models. vllm 0.6 +

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
    is_chat_api: bool = True

    def __init__(
        self,
        path: str = "",
        model: str = "",
        stream: bool = False,
        max_out_len: int = 4096,
        retry: int = 2,
        host_ip: str = "localhost",
        host_port: int = 8080,
        url: str = "",
        trust_remote_code: bool = False,
        generation_kwargs: Optional[Dict] = None,
        meta_template: Optional[Dict] = None,
        enable_ssl: bool = False,
        verbose: bool = False,
    ):
        super().__init__(
            path=path,
            stream=stream,
            max_out_len=max_out_len,
            retry=retry,
            host_ip=host_ip,
            host_port=host_port,
            url=url,
            generation_kwargs=generation_kwargs,
            meta_template=meta_template,
            enable_ssl=enable_ssl,
            verbose=verbose,
        )
        self.meta_template = (
            dict(
                round=[
                    dict(role="HUMAN", api_role="HUMAN"),
                    dict(role="BOT", api_role="BOT", generate=True),
                ],
                reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
            )
            if not meta_template
            else meta_template
        )
        self.model = model if model else self._get_service_model_path()
        self.tokenizer = None
        if path:
            self.tokenizer = AutoTokenizer.from_pretrained(path)
    
    def _get_url(self, host_ip: str, host_port: int, url: str):
        if url:
            return os.path.join(url, "chat/completions")
        base_url = self._get_base_url()
        if self.enable_ssl:
            return f"{base_url}/chat/completions"
        return f"{base_url}/chat/completions"

    def encode(self, prompt: list) -> Tuple[float, List[int]]:
        """Encode a string into tokens, measuring processing time."""
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return []
        if isinstance(prompt, list):
            messages = self.tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=False
            )
        elif isinstance(prompt, str):
            messages = prompt
        else:
            self.logger.error(f"Prompt{prompt} is not a list or string.")
            return []
        tokens = self.tokenizer.encode(messages)
        return tokens

    def decode(self, tokens: List[int]) -> Tuple[List[float], str]:
        if not self.tokenizer:
            self.logger.error("Tokenizer is not initialized.")
            return [], ""
        return self.tokenizer.decode(tokens)

    def _get_base_url(self) -> str:
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/v1"
        return f"http://{self.host_ip}:{self.host_port}/v1"

    def _get_service_model_path(self) -> str:
        base_url = self._get_base_url()
        client = OpenAI(api_key="EMPTY", base_url=base_url)
        return client.models.list().data[0].id

    async def get_request_body(
        self, input: PromptType, max_out_len: int, output: RequestOutput, **args
    ):
        if max_out_len <= 0:
            return ""
        if isinstance(input, str):
            messages = [{"role": "user", "content": input}]
        else:
            messages = []
            for item in input:
                msg = {"content": item["prompt"]}
                if item["role"] == "HUMAN":
                    msg["role"] = "user"
                elif item["role"] == "BOT":
                    msg["role"] = "assistant"
                elif item["role"] == "SYSTEM":
                    msg["role"] = "system"
                messages.append(msg)
        output.input = messages
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})

        request_body = dict(
            stream=self.stream,
            messages=messages,
        )
        if self.stream:
            request_body["stream_options"] = {"include_usage": True}
        request_body = request_body | generation_kwargs
        return request_body

    async def parse_stream_response(self, json_content, output):
        for item in json_content.get("choices", []):
            if item["delta"].get("content"):
                output.content += item["delta"]["content"]
            if item["delta"].get("reasoning_content"):
                output.reasoning_content += item["delta"]["reasoning_content"]
        if json_content.get("usage"):
            output.output_tokens = json_content["usage"]["completion_tokens"]

    async def parse_text_response(self, json_content, output):
        for item in json_content.get("choices", []):
            if content:=item["message"].get("content"):
                output.content += content
            if reasoning_content:=item["message"].get("reasoning_content"):
                output.reasoning_content += reasoning_content
        if json_content.get("usage"):
            output.output_tokens = json_content["usage"]["completion_tokens"]


@MODELS.register_module()
class VLLMCustomAPIChatStream(VLLMCustomAPIChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stream = True
