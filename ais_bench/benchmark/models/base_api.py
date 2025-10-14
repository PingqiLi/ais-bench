import sys
import json
import warnings
from abc import abstractmethod
from copy import deepcopy
import asyncio

from typing import Dict, List, Optional, Tuple, Union

from ais_bench.benchmark.utils import (
    get_logger,
)
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base import BaseModel

import aiohttp
import traceback
from ais_bench.benchmark.models.output import Output


def handle_synthetic_input(func):
    def wrapper(self, **args):
        return func(self, **args)

    return


AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=20 * 60 * 60)

PromptType = Union[PromptList, str]


class BaseAPIModel(BaseModel):
    """Base class for API model wrapper.

    Args:
        path (str): The path to the model.
        request_rate (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        traffic_cfg (ConfigDict, optional): control the request traffic rate
                "burstiness": Optional[float],    # Burstiness factor controlling interval randomness (≥0, default:0)
                "ramp_up_strategy": Optional[str],  # Ramp-up strategy type ("linear", "exponential", or None)
                "ramp_up_start_rps": Optional[float],  # Starting RPS for ramp-up (required with strategy)
                "ramp_up_end_rps": Optional[float]   # Ending RPS for ramp-up (required with strategy)
        retry (int): Number of retires if the API call fails. Defaults to 2.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        generation_kwargs (Dict, optional): The generation kwargs for the
            model. Defaults to dict().
    """

    is_api: bool = True

    def __init__(
        self,
        path: str,
        stream: bool = False,
        max_out_len: int = 2048,
        retry: int = 2,
        host_ip: str = "localhost",
        host_port: int = 8080,
        url: str = "",
        meta_template: Optional[Dict] = None,
        generation_kwargs: Dict = dict(),
        enable_ssl: bool = False,
        verbose: bool = False,
    ):
        self.logger = get_logger()
        self.path = path
        self.stream = stream
        self.max_out_len = max_out_len
        self.retry = retry
        self.meta_template = meta_template if meta_template else None
        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.url = self._get_url(host_ip, host_port, url)
        self.template_parser = APITemplateParser(self.meta_template)
        self.generation_kwargs = generation_kwargs
        self.verbose = verbose
        self.session = None

    @abstractmethod
    def _get_url(self, host_ip: str, host_port: int, url: str):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not supported"
            " to be called in base classes"
        )

    @abstractmethod
    def check_mm_prompt(self, input_data):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not supported"
            " to be called in base classes"
        )

    async def iter_lines(self, stream):
        """
        Split the input stream into lines based on multiple delimiters:
        - "\n\n" (LF LF)
        - "\r\n\r\n" (CRLF CRLF)
        - "\r\r" (CR CR)

        If the received packet does not encounter any of these delimiters,
        cache it and concatenate it with the subsequent stream.
        """
        pending = None
        async for chunk in stream:
            if pending is not None:
                chunk = pending + chunk
            lines = [
                d
                for d in chunk.replace(b"\r\n\r\n", b"\n\n")
                .replace(b"\r\r", b"\n\n")
                .split(b"\n\n")
                if d
            ]
            # If there are no lines or the chunk is empty, clear pending
            if not lines or not chunk:
                pending = None
            # If the last line's last byte matches the chunk's last byte,
            # it means the chunk did not end with '\n\n', so the last segment is incomplete
            elif lines[-1][-1] == chunk[-1]:
                pending = lines.pop()
            else:
                pending = None
            for line in lines:
                yield line
        # After the stream ends, yield any remaining incomplete data
        if pending is not None:
            yield pending

    @abstractmethod
    async def get_request_body(
        self, input_data: PromptType, max_out_len: int, output: Output, **args
    ):
        raise NotImplementedError(
            f"{self.__class__.__name__} does not supported"
            " to be called in base classes"
        )

    async def parse_text_response(self, data, output):
        raise NotImplementedError(
            f"{self.__class__.__name__} should be implemented if stream is False"
        )

    async def parse_stream_response(self, data, output):
        raise NotImplementedError(
            f"{self.__class__.__name__} should be implemented if stream is True"
        )

    async def generate(
        self,
        input_data: PromptType,
        max_out_len: int,
        output: Output,
        session: aiohttp.ClientSession = None,
        **args,
    ):
        if not session:
            self.session = aiohttp.ClientSession(
                trust_env=True, timeout=AIOHTTP_TIMEOUT
            )
            close_session = True
        else:
            self.session = session
            close_session = False
        request_body = await self.get_request_body(
            input_data, max_out_len, output, **args
        )
        retry_count = 0
        for _ in range(self.retry):
            try:
                if self.stream:
                    await self.stream_infer(request_body, output)
                else:
                    await self.text_infer(request_body, output)
                # break retry loop when request is successful
                break
            except asyncio.exceptions.CancelledError as e:
                output.success = False
                output.error_info = "Request cancelled by user"
                break
            except Exception:
                # increase retry count and set output to failed
                retry_count += 1
                output.success = False
                exc_info = sys.exc_info()
                output.error_info = (
                    f"After {retry_count} retries, request failed with exception:\n"
                    + "\n".join(traceback.format_exception(*exc_info))
                )
                await output.clear_time_points()
                continue
        if close_session:
            await self.session.close()
        return output

    async def stream_infer(self, request_body: dict, output: Output):
        headers = {"Content-Type": "application/json"}
        await output.record_time_point()
        async with self.session.post(
            url=self.url, json=request_body, headers=headers
        ) as response:
            if response.status == 200:
                async for raw_chunk in self.iter_lines(response.content):
                    chunk = raw_chunk.strip()
                    if not chunk:
                        continue
                    chunk = chunk.decode("utf-8")
                    if chunk.startswith(":"):
                        continue
                    chunk = chunk.removeprefix("data:").strip()
                    if chunk == "[DONE]":
                        break
                    await output.record_time_point()
                    data = json.loads(chunk)
                    await self.parse_stream_response(data, output)
                output.success = True
            else:
                output.error_info = response.reason
                output.success = False

    async def text_infer(self, request_body, output: Output):
        headers = {"Content-Type": "application/json"}
        await output.record_time_point()
        async with self.session.post(
            url=self.url, json=request_body, headers=headers
        ) as response:
            if response.status == 200:
                raw_data = await response.text()
                await output.record_time_point()
                data = json.loads(raw_data)
                await self.parse_text_response(data, output)
                output.success = True
            else:
                output.error_info = response.reason
                output.success = False


class APITemplateParser:
    """Intermidate prompt template parser, specifically for API models.

    Args:
        meta_template (Dict): The meta template for the model.
    """

    def __init__(self, meta_template: Optional[Dict] = None):
        self.meta_template = meta_template
        # Check meta template
        if meta_template:
            assert "round" in meta_template, "round is required in meta" " template"
            assert isinstance(meta_template["round"], list)
            keys_to_check = ["round"]

            if "reserved_roles" in meta_template:
                assert isinstance(meta_template["reserved_roles"], list)
                keys_to_check.append("reserved_roles")

            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for meta_key in keys_to_check:
                for item in meta_template[meta_key]:
                    assert isinstance(item, (str, dict))
                    if isinstance(item, dict):
                        assert (
                            item["role"] not in self.roles
                        ), "role in meta prompt must be unique!"
                        self.roles[item["role"]] = item.copy()

    def parse_template(self, prompt_template: PromptType, mode: str) -> PromptType:
        """Parse the intermidate prompt template, and wrap it with meta
        template if applicable. When the meta template is set and the input is
        a PromptList, the return value will be a PromptList containing the full
        conversation history. Each item looks like:

        .. code-block:: python

            {'role': 'user', 'prompt': '...'}).

        Args:
            prompt_template (List[PromptType]): An intermidate prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            List[PromptType]: The finalized prompt or a conversation.
        """
        assert isinstance(prompt_template, (str, list, PromptList, tuple))

        if not isinstance(prompt_template, (str, PromptList)):
            return [self.parse_template(p, mode=mode) for p in prompt_template]

        assert mode in ["ppl", "gen"]
        if isinstance(prompt_template, str):
            return prompt_template

        if self.meta_template:

            prompt = PromptList()
            # Whether to keep generating the prompt
            generate = True

            section_stack = []  # stores tuples: (section_name, start_idx)

            for i, item in enumerate(prompt_template):
                if not generate:
                    break
                if isinstance(item, str):
                    if item.strip():
                        # TODO: logger
                        warnings.warn(
                            "Non-empty string in prompt template "
                            "will be ignored in API models."
                        )
                elif isinstance(item, dict) and "section" in item:
                    if item["pos"] == "end":
                        section_name, start_idx = section_stack.pop(-1)
                        assert section_name == item["section"]
                        if section_name in ["round", "ice"]:
                            dialogue = prompt_template[start_idx:i]
                            round_ranges = self._split_rounds(
                                dialogue, self.meta_template["round"]
                            )
                            # Consider inserting multiple round examples into
                            # template
                            for i in range(len(round_ranges) - 1):
                                start = round_ranges[i]
                                end = round_ranges[i + 1]
                                round_template = dialogue[start:end]
                                role_dict = self._update_role_dict(round_template)
                                api_prompts, generate = self._prompt2api(
                                    self.meta_template["round"],
                                    role_dict,
                                    # Start generating only when the mode is in
                                    # generation and the template reaches the
                                    # last round
                                    for_gen=mode == "gen"
                                    and section_name == "round"
                                    and i == len(round_ranges) - 2,
                                )
                                prompt += api_prompts
                    elif item["pos"] == "begin":
                        assert item["section"] in ["begin", "round", "end", "ice"]
                        section_stack.append((item["section"], i + 1))
                    else:
                        raise ValueError(f'Invalid pos {item["pos"]}')
                elif section_stack[-1][0] in ["begin", "end"]:
                    role_dict = self._update_role_dict(item)
                    api_prompts, generate = self._prompt2api(
                        item, role_dict, for_gen=mode == "gen"
                    )
                    prompt.append(api_prompts)

            # merge the consecutive prompts assigned to the same role
            new_prompt = PromptList([prompt[0]])
            last_role = prompt[0]["role"]
            for item in prompt[1:]:
                if item["role"] == last_role:
                    new_prompt[-1]["prompt"] += "\n" + item["prompt"]
                else:
                    last_role = item["role"]
                    new_prompt.append(item)
            prompt = new_prompt

            if self.meta_template.get("begin", None):
                prompt.insert(0, self.meta_template["begin"])

        else:
            # in case the model does not have any meta template
            prompt = ""
            last_sep = ""
            prompt_mm = []
            for item in prompt_template:
                if isinstance(item, dict) and set(["section", "pos"]) == set(
                    item.keys()
                ):
                    continue
                if isinstance(item, str):
                    if item:
                        prompt += last_sep + item
                elif item.get("prompt", ""):
                    prompt += last_sep + item.get("prompt", "")
                elif item.get("prompt_mm", ""):
                    prompt_mm += item.get("prompt_mm", [])
                last_sep = "\n"
        return prompt if prompt else prompt_mm

    def _update_role_dict(self, prompts: Union[List, str]) -> Dict[str, Dict]:
        """Update the default role dict with the given prompts."""
        role_dict = deepcopy(self.roles)
        if isinstance(prompts, str):
            return role_dict
        elif isinstance(prompts, dict):
            prompts = [prompts]
        for prompt in prompts:
            if isinstance(prompt, dict):
                role = prompt["role"]
                if role not in self.roles:
                    role = prompt.get("fallback_role", None)
                    if not role:
                        print(
                            f"{prompt} neither has an appropriate role nor "
                            "a fallback role."
                        )
                role_dict[role].update(prompt)
        return role_dict

    def _split_rounds(
        self,
        prompt_template: List[Union[str, Dict]],
        single_round_template: List[Union[str, Dict]],
    ) -> List[int]:
        """Split the prompt template into rounds, based on single round
        template.

        Return the index ranges of each round. Specifically,
        prompt_template[res[i]:res[i+1]] represents the i-th round in the
        template.
        """
        role_idxs = {
            role_cfg["role"]: i
            for i, role_cfg in enumerate(single_round_template)
            if not isinstance(role_cfg, str)
        }
        last_role_idx = -1
        cutoff_idxs = [0]
        for idx, template in enumerate(prompt_template):
            if isinstance(template, str):
                continue
            role_idx = role_idxs.get(template["role"], None)
            if role_idx is None:
                try:
                    role_idx = role_idxs[template["fallback_role"]]
                except KeyError:
                    raise KeyError(
                        f"{template} neither has an appropriate "
                        "role nor a fallback role."
                    )
            if role_idx <= last_role_idx:
                cutoff_idxs.append(idx)
            last_role_idx = role_idx
        cutoff_idxs.append(len(prompt_template))
        return cutoff_idxs

    def _prompt2api(
        self,
        prompts: Union[List, str],
        role_dict: Dict[str, Dict],
        for_gen: bool = False,
    ) -> Tuple[List, bool]:
        """Convert the prompts to a API-style prompts, given an updated
        role_dict.

        Args:
            prompts (Union[List, str]): The prompts to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[List, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        cont = True
        if isinstance(prompts, str):
            return prompts, cont
        elif isinstance(prompts, dict):
            api_role, cont = self._role2api_role(prompts, role_dict, for_gen)
            return api_role, cont

        res = []
        for prompt in prompts:
            if isinstance(prompt, str):
                raise TypeError(
                    "Mixing str without explicit role is not " "allowed in API models!"
                )
            else:
                api_role, cont = self._role2api_role(prompt, role_dict, for_gen)
                if api_role:
                    res.append(api_role)
                if not cont:
                    break
        return res, cont

    def _role2api_role(
        self, role_prompt: Dict, role_dict: Dict[str, Dict], for_gen: bool = False
    ) -> Tuple[Dict, bool]:
        """Convert a role prompt to a string, given an updated role_dict.

        Args:
            role_prompt (Dict): The role prompt to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[Dict, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        merged_prompt = role_dict.get(
            role_prompt["role"], role_dict.get(role_prompt.get("fallback_role"))
        )
        # res_api_prompt = dict(type='', )
        if for_gen and merged_prompt.get("generate", False):
            return None, False
        res = {}
        res["role"] = merged_prompt["api_role"]
        if "prompt" in merged_prompt:
            res["prompt"] = merged_prompt.get("begin", "")
            res["prompt"] += merged_prompt.get("prompt", "")
            res["prompt"] += merged_prompt.get("end", "")
        elif "prompt_mm" in merged_prompt:
            res["prompt"] = merged_prompt.get("prompt_mm", [])
        else:
            raise ValueError("Invalid prompt content: without prompt/prompt_mm !")
        return res, True
