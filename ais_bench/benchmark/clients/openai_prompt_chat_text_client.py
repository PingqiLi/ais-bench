import json
from abc import ABC
from typing import Optional, Dict, Any

from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.utils import MiddleData
from dataclasses import dataclass, field


class OpenAIPromptChatTextClient(BaseClient, ABC):
    def __init__(self, url, retry):
        super().__init__(url, retry)
        self.model_responses_message_for_chat_history: Dict[str, Any] = field(default_factory=dict)

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
            generated_text = res['choices'][0]['message'].get('content', '')
            reasoning_content = res['choices'][0]['message'].get('reasoning_content', '')
        except Exception as e:
            raise RuntimeError(f"Process response failed and the reason is {e}")
        self.model_responses_message_for_chat_history = res["choices"][0]["message"]
        if "</think>" in generated_text:
            parts = generated_text.split("</think>")
            reasoning_content= parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
            inputs.output = parts[-1].lstrip("\n")
            self.model_responses_message_for_chat_history['content'] = inputs.output
            self.model_responses_message_for_chat_history['reasoning_content'] = reasoning_content
        else:
            inputs.output = generated_text