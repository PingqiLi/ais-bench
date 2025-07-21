from abc import ABC

from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.utils import MiddleData


class VLLMTextClient(BaseClient, ABC):
    def construct_request_body(
        self,
        inputs: str,
        parameters: dict = None,
    ) -> dict:
        data = dict(prompt=inputs, stream = False)
        data = data | parameters
        return data

    def update_middle_data(self, res: dict, inputs: MiddleData):
        generated_text = res['text'][0]
        if generated_text:
            inputs.output = generated_text
            inputs.num_generated_chars = len(generated_text)
