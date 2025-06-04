from abc import ABC
import json

from ais_bench.benchmark.clients.base_client import BaseStreamClient
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.registry import CLIENTS


@CLIENTS.register_module()
class OpenAIChatStreamClient(BaseStreamClient, ABC):
    def preprocess_cur_line(self, cur_line: str) -> str:
        if "\ndata" in cur_line:
            end_ix = cur_line.find("data: [DONE]")
            cur_line = cur_line if end_ix < 0 else cur_line[:end_ix]
            data_blocks = cur_line.strip().split('\n\n')
            merged_data = None
            for block in data_blocks:
                # 去掉 "data: " 前缀并解析 JSON
                json_str = block.replace('data: ', '')
                data = json.loads(json_str)

                # 如果是第一条数据，初始化 merged_data
                if merged_data is None:
                    merged_data = data
                else:
                    # 合并 choices
                    merged_data['choices'].extend(data['choices'])
            return json.dumps(merged_data)
        else:
            end_ix = cur_line.find("data: [DONE]")
            return cur_line if end_ix < 0 else cur_line[:end_ix]

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
        return data

    def process_stream_line(self, json_content: dict) -> dict:
        response = {}
        generated_text = ""
        for item in json_content["choices"]:
            generated_text += item["delta"]["content"]
        if generated_text:
            response.update({"generated_text": generated_text})
        if self.do_performance:
            response.update({"token_str": generated_text})
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
        chunk_time_point = res.get("chunk_time_point")
        if chunk_time_point:
            inputs.chunk_time_point_list.append(chunk_time_point)
        inputs.num_generated_tokens += 1
        return generated_text
