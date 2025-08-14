from abc import ABC

from ais_bench.benchmark.clients.base_client import BaseStreamClient
from ais_bench.benchmark.utils import MiddleData


class MindieStreamClient(BaseStreamClient, ABC):
    def construct_request_body(
        self,
        inputs: str,
        parameters: dict = None,
    ) -> dict:
        return dict(inputs=inputs, stream=True, parameters=parameters)

    def process_stream_line(self, json_content: dict) -> dict:
        response = {}
        generated_text = json_content.get("generated_text", None)
        if generated_text:
            response.update({"generated_text": generated_text})
        if self.do_performance:
            response.update({"token_str": json_content["token"]["text"]})
        return response

    def update_middle_data(self, res: dict, inputs: MiddleData):
        generated_text = res.get("generated_text", "")
        if generated_text:
            inputs.output += generated_text
            inputs.num_generated_chars = len(generated_text)
        prefill_time = res.get("prefill_time")
        if prefill_time:
            inputs.prefill_latency = prefill_time
        decode_time = res.get("decode_time")
        if decode_time:
            inputs.decode_cost.append(decode_time)
