from abc import ABC

import uuid
from ais_bench.benchmark.clients.base_client import BaseStreamClient
from ais_bench.benchmark.utils import MiddleData


class OpenAIStreamClient(BaseStreamClient, ABC):

    def construct_request_body(
        self,
        inputs: str,
        parameters: dict = None,
    ) -> dict:
        data = dict(
            prompt = inputs,
            stream = True,
        )
        data = data | parameters
        data["stream_options"] = {"include_usage": True}
        return data

    def process_stream_line(self, json_content: dict) -> dict:
        response = {}
        generated_text = ""
        if len(json_content.get('choices', [])) > 0:
            generated_text = json_content['choices'][0]['text']
        if generated_text:
            response.update({"generated_text": generated_text})
        if self.do_performance:
            response.update({"token_str": generated_text})
        if json_content.get("usage"):
            response.update({"completion_tokens": json_content["usage"]["completion_tokens"]})
        return response

    def update_middle_data(self, res: dict, inputs: MiddleData):
        generated_text = res.get("generated_text", "")
        if generated_text:
            inputs.output += generated_text
            inputs.num_generated_chars = len(inputs.output)
        prefill_time = res.get("prefill_time")
        if prefill_time:
            inputs.prefill_latency = prefill_time
        decode_time = res.get("decode_time")
        if decode_time:
            inputs.decode_cost.append(decode_time)
        if res.get("completion_tokens"):
            inputs.num_generated_tokens = res.get("completion_tokens")
