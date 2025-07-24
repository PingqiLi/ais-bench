import json
import time
import re
from abc import abstractmethod, ABC

from ais_bench.benchmark.clients.base_client import BaseStreamClient, _stream_data_split
from ais_bench.benchmark.utils import MiddleData
from ais_bench.benchmark.registry import CLIENTS
from ais_bench.benchmark.utils.valid_global_consts import valid_max_chunk_size


@CLIENTS.register_module()
class OpenAIChatStreamSglangClient(BaseStreamClient, ABC):
    def preprocess_cur_line(self, cur_line: str) -> str:
        if "\ndata" in cur_line:
            end_ix = cur_line.find("data: [DONE]")
            cur_line = cur_line if end_ix < 0 else cur_line[:end_ix]
            data_blocks = cur_line.strip().split('\n\n')
            print(f"{data_blocks=}")
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
                if data.get("usage"):
                    merged_data["usage"] = data["usage"]
            print(f"{merged_data=}")
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
        data["stream_options"] = {"include_usage": True}
        return data

    def process_stream_line(self, json_content: dict) -> dict:
        response = {}
        generated_text = ""
        reasoning_content = ""
        for item in json_content.get("choices", []):
            if item["delta"].get("content"):  # content maybe null in sglang service
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
        if generated_text:
            inputs.output += generated_text
            inputs.num_generated_chars = len(inputs.output)
        if reasoning_content:
            inputs.output_reasoning += reasoning_content
            inputs.num_generated_chars += len(reasoning_content)
        prefill_time = res.get("prefill_time")
        if prefill_time:
            inputs.prefill_latency = prefill_time
        decode_time = res.get("decode_time")
        if decode_time:
            inputs.decode_cost.append(decode_time)
        chunk_time_point = res.get("chunk_time_point")
        if chunk_time_point:
            inputs.chunk_time_point_list.append(chunk_time_point)
        if res.get("completion_tokens"):
            inputs.num_generated_tokens = res.get("completion_tokens")
    
    def process_response(self, response, last_time_point):
        time_name = "prefill_time"
        for byte_line in response.stream(amt=valid_max_chunk_size()):
            
            if byte_line == b"\n":
                print(f"{byte_line=}")
                continue
            cur_line = self.preprocess_cur_line(byte_line.decode())
            try:
                for json_content in _stream_data_split(cur_line):
                    cur_time_point = time.perf_counter()
                    response_dict = self.process_stream_line(json_content)
                    if not response_dict.get("generated_text") and not response_dict.get("completion_tokens"):  #first return chunk: None, reset start time
                        continue 
                    if time_name not in response_dict.keys():
                        response_dict[time_name] = (
                            cur_time_point - last_time_point
                        ) * 1000
                        response_dict["chunk_time_point"] = cur_time_point * 1000
                    yield response_dict
                    time_name = "decode_time"
                    last_time_point = time.perf_counter()
            except Exception as error:
                raise ValueError(f"[StreamResponseError] {error}! Raw server response: {cur_line}")