from abc import ABC
import json

from ais_bench.benchmark.clients.base_client import BaseStreamClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.registry import CLIENTS


@CLIENTS.register_module()
class OpenAIChatStreamClient(BaseStreamClient, ABC):

    def construct_request_body(
        self,
        inputs: list,
        parameters: dict = None,
    ) -> dict:
        data = dict(
            stream = True,
            messages = inputs,
        )
        data = data | parameters
        data["stream_options"] = {"include_usage": True}
        return data

    def process_stream_line(self, json_content: dict) -> dict:
        response = {}
        generated_text = ""
        reasoning_content = ""
        for item in json_content.get("choices", []):
            if item["delta"].get("content"):
                generated_text += item["delta"]["content"]
            if item["delta"].get("reasoning_content"):
                reasoning_content += item["delta"]["reasoning_content"]
        if generated_text:
            response.update({"generated_text": generated_text})
        if reasoning_content:
            response.update({"reasoning_content": reasoning_content})
        if self.do_performance:
            response.update({"token_str": generated_text})
        if json_content.get("usage"):
            response.update({"completion_tokens": json_content["usage"]["completion_tokens"]})
        return response

    def update_middle_data(self, res: dict, inputs: MiddleData):
        generated_text = res.get("generated_text", "")
        reasoning_content = res.get("reasoning_content", "")
        if reasoning_content:
            inputs.output_reasoning += reasoning_content
            inputs.num_generated_chars += len(reasoning_content)
        if generated_text:
            inputs.output += generated_text
            inputs.num_generated_chars += len(generated_text)
        prefill_time = res.get("prefill_time")
        if prefill_time:
            inputs.prefill_latency = prefill_time
        decode_time = res.get("decode_time")
        if decode_time:
            inputs.decode_cost.append(decode_time)
        if res.get("completion_tokens"):
            inputs.num_generated_tokens = res.get("completion_tokens")
