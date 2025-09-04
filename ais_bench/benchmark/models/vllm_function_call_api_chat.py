import os
import uuid
import json
from typing import Any, Dict, List, Optional, Union, Tuple
from mmengine.config import ConfigDict

from openai import OpenAI

from ais_bench.benchmark.registry import MODELS
from ais_bench.benchmark.utils.prompt import PromptList

from ais_bench.benchmark.models.base_api import handle_synthetic_input
from ais_bench.benchmark.models.performance_api import PerformanceAPIModel
from ais_bench.benchmark.clients import (
    OpenAIChatTextClient,
    OpenAIFunctionChatTextClient,
    OpenAIPromptChatTextClient,
)
from ais_bench.benchmark.utils.results import MiddleData
from ais_bench.benchmark.utils.build import build_client_from_cfg
from ais_bench.benchmark.datasets.bfcl.bfcl_dependency import *

PromptType = Union[PromptList, str, dict]


class VLLMFunctionBaseAPIChat(PerformanceAPIModel):
    """
    Base class for VLLM function-calling chat APIs, providing standard interfaces for
    pre-processing queries, injecting holdout functions, handling multi-turn inference,
    and recording execution results.
    """

    def __init__(self, client):
        """
        Initialize the API wrapper.

        Args:
            client: An instance of the VLLM client or wrapper used to perform requests.
        """
        self.client = client

    def pre_query_processing(self, input: dict) -> dict:
        """
        Prepare and sanitize the user input before sending to the model.

        Args:
            input (dict): Raw input payload from the caller.

        Returns:
            dict: Processed payload ready for inference.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement pre_query_processing method."
        )

    def add_holdout_function(
        self, input: dict, inference_data: dict, holdout_function: list[dict]
    ):
        """
        Inject or configure additional functions that the model should consider but not execute
        immediately (holdout functions).

        Args:
            input (dict): Original or pre-processed input payload.
            inference_data (dict): Context or metadata collected during inference.
            holdout_function (list[dict]): List of function definitions to hold out.

        Returns:
            None: Modifies inference_data or input in-place to include holdout definitions.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement add_holdout_function method."
        )

    def inference_multi_turn(
        self, cache_data, generation_kwargs, inference_data, current_turn_response
    ):
        """
        Perform or accumulate results across multiple chat turns.

        Args:
            cache_data: Persistent state between turns (e.g., message log).
            generation_kwargs (dict): Parameters for the model call (e.g. temperature).
            inference_data (dict): Current turn context and metadata.
            current_turn_response: Response object from the last API call.

        Returns:
            Updated cache_data or aggregated output.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement inference_multi_turn method."
        )

    def add_execution_results(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """
        Record the actual execution results after function calling.

        Args:
            inference_data (dict): Context from the inference stage.
            execution_results (list[str]): Outputs returned by executing functions.
            model_response_data (dict): Raw model API response for reference.

        Returns:
            dict: Enriched inference_data containing execution outputs.

        Raises:
            NotImplementedError: Must be overridden in subclass.
        """
        raise NotImplementedError(
            "Subclasses must implement add_execution_results method."
        )

    def _add_assistant_message(
        self, inference_data: dict, model_responses_message_for_chat_history
    ) -> dict:
        inference_data["message"].append(model_responses_message_for_chat_history)
        return inference_data

    def _get_test_category(self, data_name: str) -> str:
        """Extract test category from data_name."""
        return data_name.rsplit("_", 1)[0]

    def _load_json_field(self, input: dict, key: str):
        """Safely load a JSON field from input dict."""
        return json.loads(input[key]) if key in input else []


class VLLMFunctionAPIChat(VLLMFunctionBaseAPIChat):
    def __init__(self, client):
        super().__init__(client)

    def pre_query_processing(self, input: dict) -> dict:
        """Preprocess inputs and compile tool information."""
        inference_data = {"message": []}
        functions = self._load_json_field(input, "function")
        test_category = self._get_test_category(input.get("data_name", ""))
        inference_data = self._compile_tools(inference_data, functions, test_category)
        input["prompt"] = self._load_json_field(input, "prompt")
        return inference_data

    def add_holdout_function(
        self, input: dict, inference_data: dict, holdout_function: list[dict]
    ):
        functions = self._load_json_field(input, "function")
        test_category = self._get_test_category(input.get("data_name", ""))
        functions.extend(holdout_function)
        inference_data = self._compile_tools(inference_data, functions, test_category)
        current_turn_message = [
            {
                "role": "user",
                "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
            }
        ]
        return inference_data, current_turn_message

    def inference_multi_turn(
        self, cache_data, generation_kwargs, inference_data, current_turn_response
    ):
        generation_kwargs.update({"tools": inference_data.get("tools")})
        response = self.client.request(cache_data, generation_kwargs)
        inference_data["tool_call_ids"] = self.client.tool_call_ids
        current_turn_response.append(response)
        inference_data = self._add_assistant_message(
            inference_data, self.client.model_responses_message_for_chat_history
        )
        try:
            result = json.loads(response)
            result = convert_to_function_call(result)
        except Exception as e:
            return []
        return result

    def _compile_tools(self, inference_data: dict, functions: dict, test_category: str) -> dict:
        """编译函数为工具格式。"""
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = convert_to_tool(functions, GORILLA_TO_OPENAPI, ModelStyle.OpenAI)
        inference_data["tools"] = tools
        return inference_data

    def add_execution_results(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response_data: dict,
    ) -> dict:
        """将执行结果添加为 tool 消消息。"""
        for execution_result, tool_call_id in zip(
            execution_results, self.client.tool_call_ids
        ):
            tool_message = {
                "role": "tool",
                "content": execution_result,
                "tool_call_id": tool_call_id,
            }
            inference_data["message"].append(tool_message)
        return inference_data


class VLLMPromptAPIChat(VLLMFunctionBaseAPIChat):
    def __init__(self, client):
        super().__init__(client)

    def pre_query_processing(self, input: dict) -> dict:
        """预处理输入，处理系统提示。"""
        functions = self._load_json_field(input, "function")
        test_category = self._get_test_category(input["data_name"])
        functions = func_doc_language_specific_pre_processing(functions, test_category)
        prompts = self._load_json_field(input, "prompt")
        prompts[0] = system_prompt_pre_processing_chat_model(
            prompts[0], functions, test_category
        )
        input["prompt"] = prompts
        return {"message": []}

    def add_holdout_function(
        self, input: dict, inference_data: dict, holdout_function: list[dict]
    ):
        current_turn_message = [
            {
                "role": "user",
                "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_PROMPTING.format(
                    functions=holdout_function
                ),
            }
        ]
        return inference_data, current_turn_message

    def inference_multi_turn(
        self, cache_data, generation_kwargs, inference_data, current_turn_response
    ):
        response = self.client.request(cache_data, generation_kwargs)
        current_turn_response.append(cache_data.output)
        inference_data = self._add_assistant_message(
            inference_data, self.client.model_responses_message_for_chat_history
        )
        try:
            result = default_decode_execute_prompting(cache_data.output)
        except Exception:
            return []
        return result

    def add_execution_results(
        self,
        inference_data: dict,
        execution_results: list[str],
        model_response: list[str],
    ) -> dict:
        model_response_data = {"model_responses_decoded": model_response}
        formatted_results_message = format_execution_results_prompting(
            {}, execution_results, model_response_data
        )
        inference_data["message"].append(
            {"role": "user", "content": formatted_results_message}
        )
        return inference_data


@MODELS.register_module()
class VLLMFunctionCallAPIChat(PerformanceAPIModel):
    def __init__(
        self,
        path: str = "",
        model: str = "",
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
        custom_client=dict(type=OpenAIFunctionChatTextClient),
        generation_kwargs: Optional[Dict] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            meta_template=meta_template,
            request_rate=request_rate,
            traffic_cfg=traffic_cfg,
            rpm_verbose=rpm_verbose,
            retry=retry,
            verbose=verbose,
            generation_kwargs=generation_kwargs,
            trust_remote_code=trust_remote_code,
        )
        if not BFCL_INSTALLED:
            raise ImportError(
                "Missing required package 'bfcl-eval'. To run BFCL evaluation, "
                "install it via: pip3 install -r requirements/bfcl_dependencies.txt --no-deps"
            )
        self.host_ip = host_ip
        self.host_port = host_port
        self.enable_ssl = enable_ssl
        self.base_url = self._get_base_url()
        self.endpoint_url = os.path.join(self.base_url, "chat/completions")
        self.model = model if model else self._get_service_model_path()
        self.is_multi_modal = False
        self.returns_tool_calls = kwargs.pop("returns_tool_calls", False)
        self.model_name = "function-call-model-" + str(uuid.uuid4()).split("-")[-1]
        self.init_client(custom_client)
        if self.returns_tool_calls:
            self.impl = VLLMFunctionAPIChat(self.client)
        else:
            self.impl = VLLMPromptAPIChat(self.client)

    def init_client(self, custom_client):
        if not self.returns_tool_calls:
            custom_client = dict(type=OpenAIPromptChatTextClient)
        if not isinstance(custom_client, dict):
            self.logger.warning(
                f"Value of custom_client: {custom_client} is not a dict! Use Default"
            )
            custom_client = dict(type=OpenAIFunctionChatTextClient)
        custom_client["url"] = self.endpoint_url
        custom_client["retry"] = self.retry
        self.client = build_client_from_cfg(custom_client)

    @handle_synthetic_input
    def _generate(self, input: PromptType, max_out_len: int) -> str:
        generation_kwargs = self.generation_kwargs.copy()
        generation_kwargs.update({"max_tokens": max_out_len})
        generation_kwargs.update({"model": self.model})
        if "multi_turn" in input.get("data_name"):
            return self._inference_multi_turn(input, generation_kwargs)
        else:
            return self._inference_single_turn(input, generation_kwargs)

    def _add_next_turn_user_message(
        self, inference_data: dict, user_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(user_message)
        return inference_data

    def _inference_multi_turn(self, input: PromptType, generation_kwargs: Dict) -> str:
        data_id: str = input.get("data_id")
        test_entry_id = input.get("data_name")
        test_category: str = test_entry_id.rsplit("_", 1)[0]
        initial_config = input["initial_config"]
        involved_classes = input["involved_classes"]
        initial_config = json.loads(initial_config)
        involved_classes = json.loads(involved_classes)
        holdout_function: dict[int, list] = json.loads(input.get("missed_function", '{}'))

        force_quit = False
        all_model_response: list[list] = []
        inference_data: dict = self.impl.pre_query_processing(input)
        all_multi_turn_messages = input.get("prompt", [])

        for turn_idx, current_turn_message in enumerate(all_multi_turn_messages):
            current_turn_message: list[dict]
            if str(turn_idx) in holdout_function:
                inference_data, current_turn_message = self.impl.add_holdout_function(
                    input, inference_data, holdout_function[str(turn_idx)]
                )

            inference_data = self._add_next_turn_user_message(
                inference_data, current_turn_message
            )

            current_turn_response = []

            cache_data = self.prepare_input_data(inference_data.get("message"), data_id)
            count = 0
            while True:
                decoded_model_responses = self.impl.inference_multi_turn(
                    cache_data, generation_kwargs, inference_data, current_turn_response
                )
                if is_empty_execute_response(decoded_model_responses):
                    break

                execution_results, involved_instances = execute_multi_turn_func_call(
                    decoded_model_responses,
                    initial_config,
                    involved_classes,
                    self.model_name,
                    test_entry_id,
                    long_context=(
                        "long_context" in test_category or "composite" in test_category
                    ),
                )
                inference_data = self.impl.add_execution_results(
                    inference_data, execution_results, decoded_model_responses
                )

                count += 1
                if count > MAXIMUM_STEP_LIMIT:
                    force_quit = True
                    break
            all_model_response.append(current_turn_response)
            if force_quit:
                break
        if all_model_response:
            all_model_response_str = json.dumps(all_model_response)
            cache_data.output = all_model_response_str
        self.set_result(cache_data)
        return cache_data.output

    def _inference_single_turn(self, input: PromptType, generation_kwargs: Dict) -> str:

        data_id: str = input.get("data_id")

        inference_data = self.impl.pre_query_processing(input)
        inference_data = self._add_next_turn_user_message(
            inference_data, input["prompt"][0]
        )
        if self.returns_tool_calls:
            generation_kwargs.update({"tools": inference_data.get("tools")})
        cache_data = self.prepare_input_data(inference_data["message"], data_id)

        response = self.client.request(cache_data, generation_kwargs)
        self.set_result(cache_data)
        return "".join(response)

    def _get_base_url(self):
        if self.enable_ssl:
            return f"https://{self.host_ip}:{self.host_port}/v1"
        return f"http://{self.host_ip}:{self.host_port}/v1"

    def _get_service_model_path(self):
        client = OpenAI(api_key="EMPTY", base_url=self.base_url)
        return client.models.list().data[0].id

    def prepare_input_data(self, inputs: list, data_id: int = -1) -> MiddleData:
        """Prepare input data, tokenize if performance mode is enabled."""
        rrid = uuid.uuid4().hex
        cache_data = self.result_cache[rrid]
        cache_data.data_id = data_id
        cache_data.request_id = rrid
        cache_data.input_data = inputs
        return cache_data