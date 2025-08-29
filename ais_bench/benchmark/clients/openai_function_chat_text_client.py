import json
from abc import ABC
from typing import Optional, Dict, Any

from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.utils import MiddleData
from dataclasses import dataclass, field


class OpenAIFunctionChatTextClient(BaseClient, ABC):
    def __init__(self, url, retry):
        super().__init__(url, retry)
        self.model_responses_message_for_chat_history: Dict[str, Any] = field(default_factory=dict)
        self.tool_call_ids = []
        
    def construct_request_body(
        self,
        inputs: dict,
        parameters: dict = None,
    ) -> dict:
        data = dict(
            messages = inputs,
            stream = False,
        )
        data = data | parameters
        return data

    def update_middle_data(self, res: dict, inputs: MiddleData):
        try:
            tool_calls = res["choices"][0]["message"].get("tool_calls", [])
            if tool_calls is None:
                tool_calls = []
            model_responses = [
                {func_call["function"]["name"].strip(): func_call["function"]["arguments"]}
                for func_call in tool_calls
            ]
            self.tool_call_ids = [
                func_call["id"] for func_call in tool_calls
            ]
            model_responses = json.dumps(model_responses)
            if not model_responses:
                model_responses = res["choices"][0]["message"].get("content", "")
        except Exception as e:
            raise RuntimeError(f"Process response failed and the reason is {e}")
        self.model_responses_message_for_chat_history = res["choices"][0]["message"]
        
        if model_responses:
            inputs.output = model_responses
            inputs.num_generated_chars = len(model_responses)