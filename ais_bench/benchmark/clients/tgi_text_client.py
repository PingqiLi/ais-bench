from abc import ABC

from ais_bench.benchmark.clients.base_client import BaseClient
from ais_bench.benchmark.utils import MiddleData


class TGITextClient(BaseClient, ABC):
    def construct_request_body(
        self,
        inputs: str,
        parameters: dict = None,
    ) -> dict:
        return dict(inputs=inputs, parameters=parameters)

    def update_middle_data(self, res: dict, inputs: MiddleData):
        try:
            generated_text = res["generated_text"]
        except Exception as e:
            raise RuntimeError(f"Process response failed and the reason is {e}")
        if generated_text:
            inputs.output = generated_text
            inputs.num_generated_chars = len(generated_text)
