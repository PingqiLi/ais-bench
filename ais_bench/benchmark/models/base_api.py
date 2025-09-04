import os
import re
import sys
import threading
import time
import json
import tempfile
import warnings
import queue
from tqdm import tqdm
from abc import abstractmethod
from copy import deepcopy
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed
from time import sleep
from typing import Dict, List, Optional, Tuple, Union, Any
from mmengine.config import ConfigDict

from ais_bench.benchmark.utils import get_logger, PRESSURE_TIME_MAX, PRESSURE_TIME_MIN, CONNECTION_ADD_RATE_MIN
from ais_bench.benchmark.global_consts import PRESSURE_TIME, CONNECTION_ADD_RATE
from ais_bench.benchmark.utils.prompt import PromptList, is_mm_prompt

from .base import BaseModel


PromptType = Union[PromptList, str, dict]


class BaseAPIModel(BaseModel):
    """Base class for API model wrapper.

    Args:
        path (str): The path to the model.
        request_rate (int): The maximum queries allowed per second
            between two consecutive calls of the API. Defaults to 1.
        traffic_cfg (ConfigDict, optional): control the request traffic rate 
                "burstiness": Optional[float],    # Burstiness factor controlling interval randomness (≥0, default:0)
                "ramp_up_strategy": Optional[str],  # Ramp-up strategy type ("linear", "exponential", or None)
                "ramp_up_start_rps": Optional[float],  # Starting RPS for ramp-up (required with strategy)
                "ramp_up_end_rps": Optional[float]   # Ending RPS for ramp-up (required with strategy)
        retry (int): Number of retires if the API call fails. Defaults to 2.
        max_seq_len (int): The maximum sequence length of the model. Defaults
            to 2048.
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        generation_kwargs (Dict, optional): The generation kwargs for the
            model. Defaults to dict().
    """

    is_api: bool = True

    def __init__(self,
                 path: str,
                 request_rate: int = 1,
                 traffic_cfg: Optional[ConfigDict] = None,
                 rpm_verbose: bool = False,
                 retry: int = 2,
                 max_seq_len: int = 2048,
                 meta_template: Optional[Dict] = None,
                 generation_kwargs: Dict = dict(),
                 verbose: bool = False):
        self.logger = get_logger()
        self.path = path
        self.tqdm_pos = -1
        self.max_seq_len = max_seq_len
        if hasattr(self, "is_chat_api") and self.is_chat_api:
            self.meta_template = dict(
                round=[
                    dict(role="HUMAN", api_role="HUMAN"),
                    dict(role="BOT", api_role="BOT", generate=True),
                ],
                reserved_roles=[dict(role="SYSTEM", api_role="SYSTEM")],
            )
        else:
            self.meta_template = meta_template
        self.retry = retry
        self.rpm_verbose = rpm_verbose
        self.request_rate = request_rate
        self.traffic_cfg = traffic_cfg
        self.token_bucket = None
        self.template_parser = APITemplateParser(self.meta_template)
        self.generation_kwargs = generation_kwargs
        self.verbose = verbose
        self.lock = threading.Lock()
        self.result_cache = []
        self.post_time = 0
        self.rec_time = 0
        self.start_time = 0
        self.log_per_request = 10
        self.tmp_result_queue = Queue(-1)
        self.post_req_num = 0
        self.get_req_num = 0
        self.failed_num = 0
        self.finish_num = 0
        self.task_finish = False
        self.interrupted = False

    @abstractmethod
    def generate(self, inputs: List[PromptType],
                 max_out_len: int) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings or PromptDicts.
                The PromptDict should be organized in AISBench'
                API format.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support'
                                  ' gen-based evaluation yet, try ppl-based '
                                  'instead.')

    def delay_generate(self):
        if hasattr(self, 'tokens') and self.tokens:
            self.tokens.acquire()

    def set_description(self, p_log):
        num_width = max(
            len(str(self.get_req_num)),
            len(str(self.failed_num)),
            5
        ) + 2

        rec_time = max(0, self.rec_time - self.start_time)
        post_time = max(0, self.post_time - self.start_time)
        time_width = len(str(rec_time).split('.'))
        left_num = num_width + 2
        left_time = time_width + 2

        desc = (
            f"{'Pid:':<{left_num}}{os.getpid():>{num_width}d} | "
            f"{'Post:':<{left_num}}{self.post_req_num:>{num_width}d} | "
            f"{'Received:':<{left_num}}{self.get_req_num:>{num_width}d} | "
            f"{'Failed:':<{left_num}}{self.failed_num:>{num_width}d} | "
            f"{'Post Time:':<{left_time}}{post_time:>{time_width}.2f}s | "
            f"{'Receive Time:':<{left_time}}{rec_time:>{time_width}.2f}s"
        )
        p_log.set_description(desc)
        
    def update_count(self):
        while not self.client.request_counter.empty():
            try:
                req_type = self.client.request_counter.get_nowait()
                if req_type == "post_req":
                    self.post_req_num += 1
                elif req_type == "get_req":
                    self.get_req_num += 1
                elif req_type == "finish_req":
                    self.finish_num += 1
                else:
                    self.failed_num += 1
            except queue.Empty:
                # Queue is empty, break the inner loop
                break

    def draw_plog(self, ori_nums, length, pos):
        p_log = tqdm(total=0, desc="", position=3 * pos + 1, bar_format="{desc}", leave=True)
        with tqdm(total=ori_nums, initial=ori_nums - length, desc=f"Process-{pos} pid:{str(os.getpid())}", delay=0.01, position=3 * pos , ascii=False) as pbar:
            prev = 0
            pre_post = 0
            pre_get = 0
            pre_failed = 0
            pbar.refresh()  # Force immediate display of progress bar
            while True:
                self.update_count()
                # Process all available items in the queue before continuing
                if self.post_req_num != pre_post:
                    pre_post = self.post_req_num
                    if pre_post % self.log_per_request == 0 or pre_post >= length:
                        self.post_time = time.perf_counter()
                        self.set_description(p_log)
                if self.get_req_num != pre_get or pre_failed != self.failed_num:
                    pre_get = self.get_req_num
                    pre_failed = self.failed_num
                cur = self.failed_num + self.finish_num
                if cur != prev:
                    pbar.update(cur - prev)
                    prev = cur
                    if cur % self.log_per_request == 0 :
                        self.rec_time = time.perf_counter()
                        self.set_description(p_log)
                if cur >= length or self.task_finish:
                    self.rec_time = time.perf_counter()
                    self.set_description(p_log)
                    break
                time.sleep(0.1)
        p_log.close()

    def draw_plog_pressure(self, pos):
        p_log = tqdm(total=0, desc="", position=3 * pos + 1, bar_format="{desc}", leave=True)
        prev = 0
        pre_post = 0
        pre_get = 0
        pre_failed = 0
        while True:
            # Process all available items in the queue before continuing
            self.update_count()
            if self.post_req_num != pre_post:
                pre_post = self.post_req_num
                if pre_post % self.log_per_request == 0:
                    self.post_time = time.perf_counter()
                    self.set_description(p_log)
            if self.get_req_num != pre_get or pre_failed != self.failed_num:
                pre_get = self.get_req_num
                pre_failed = self.failed_num
            cur = self.failed_num + self.get_req_num
            if cur != prev:
                prev = cur
                if cur % self.log_per_request == 0 :
                    self.rec_time = time.perf_counter()
                    self.set_description(p_log)
            if self.task_finish:
                self.rec_time = time.perf_counter()
                self.set_description(p_log)
                break
            time.sleep(0.1)

    def save_tmp_result(self, tmp_result_json_path):
        """Read tmp cache queue to save inference results in real time."""
        res_cache = dict()
        cur_res_num = 0
        def update(tmp_res):
            data_id = str(tmp_res.get("data_id", "-1"))
            res_cache[data_id] = dict(
                origin_prompt=tmp_res.get("origin_prompt"),
                prediction=tmp_res.get("prediction"),
            )
            if tmp_res.get('gold'):
                res_cache[data_id]['gold'] = tmp_res.get('gold')

        while True:
            tmp_res = self.tmp_result_queue.get()
            if not tmp_res or self.task_finish:
                while tmp_res:
                    update(tmp_res)
                    tmp_res = self.tmp_result_queue.get()
                self._atomic_dump(res_cache, tmp_result_json_path)
                break
            update(tmp_res)
            cur_res_num += 1
            if cur_res_num % self.log_per_request == 0:
                self._atomic_dump(res_cache, tmp_result_json_path)

    def _atomic_dump(self, data: dict, target_path: str):
        """Atomically write JSON data to target_path."""
        dir_name = os.path.dirname(target_path) or '.'
        # Write to temporary file in same directory
        with tempfile.NamedTemporaryFile('w', dir=dir_name, delete=False, encoding='utf-8') as tf:
            json.dump(data, tf, indent=4, ensure_ascii=False)
            tf.flush()
            os.fsync(tf.fileno())
            temp_path = tf.name
        # Replace target file atomically
        os.replace(temp_path, target_path)

    def generate_from_queue(
        self,
        data_queue: Any,
        **extra_gen_kwargs: Any
    ) -> None:
        """Consume items from a queue and generate outputs concurrently, with optional rate limiting and progress logging.

        Args:
            data_queue (Queue): A thread-safe queue supplying input data items.
            pos (int, optional): Position index for the progress bar display.
            max_out_len (int, optional): Maximum length of each generated output (default: 1).
            concurrency (int): Number of worker threads in the pool.
            data_nums (int): Total expected number of items to process (for logging).
            sleep_offsets (list[float]): the sleep interval offsets for each request; if empty, no rate limiting is applied.
            rpm_verbose (bool, optional): Verbosity flag for rate limiter logging (in TokenBucket).

        Returns:
            None: This method runs until the queue is exhausted and then exits.
        """
        self.tqdm_pos = extra_gen_kwargs.get("process_id")
        tmp_result_dir = extra_gen_kwargs.get('tmp_result_dir')
        max_out_len = extra_gen_kwargs.get("max_out_len", 1)
        concurrency = extra_gen_kwargs.get("concurrency")
        ori_nums = extra_gen_kwargs.get("ori_nums")
        data_nums = extra_gen_kwargs.get("data_nums")
        self.log_per_request = max(1, int((ori_nums - data_nums) / 0.01))

        sleep_offsets = extra_gen_kwargs.get("sleep_offsets", [])
        global_start_time = extra_gen_kwargs.get("global_start_time", time.perf_counter())

        if sleep_offsets:
            self.token_bucket = TokenBucket(
                sleep_intervals=sleep_offsets, 
                verbose=self.rpm_verbose,
                start_time=global_start_time
            )
            self.logger.info(f"Process {self.tqdm_pos} using precomputed sleep offsets "
                       f"with {len(sleep_offsets)} requests")
        else:
            self.token_bucket = None

        if (not hasattr(self, "do_performance")) or (not self.do_performance):
            os.makedirs(tmp_result_dir, exist_ok=True)
            tmp_result_json_path = os.path.join(tmp_result_dir, f"tmp_{self.tqdm_pos}_{os.getpid()}_{str(time.time()).split('.')[0]}.json")
            tmp_save_thread = threading.Thread(target=self.save_tmp_result, args=(tmp_result_json_path,), daemon=True)
            tmp_save_thread.start()
        else:
            tmp_save_thread = None
        draw_thread = threading.Thread(target=self.draw_plog, args=(ori_nums, data_nums, self.tqdm_pos,), daemon=True)
        draw_thread.start()
        pool_size = concurrency
        self.start_time = time.perf_counter() if not global_start_time else global_start_time
        self.futures = []
        try:
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                input_data = data_queue.get()
                while input_data is not None:
                    future = executor.submit(self._generate, input_data, input_data.get("max_tokens", max_out_len))
                    self.futures.append(future)
                    input_data = data_queue.get()
                data_queue.put(None)
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user (Ctrl+C).")
            self.interrupted = True
            # cancel unfinished tasks
            for future in self.futures:
                if not future.done():
                    future.cancel()
            # force shutdown thread pool
            executor.shutdown(wait=False)
            # clear data queue
            while not data_queue.empty():
                try:
                    data_queue.get_nowait()
                except:
                    break
        except Exception as e:
            self.logger.error(f"Infer task end because error: {e}")
        finally:
            self.task_finish = True
            self.tmp_result_queue.put(None)
            draw_thread.join()
            if tmp_save_thread:
                tmp_save_thread.join()

    def pressure_generate_from_queue(
        self,
        shared_inputs: Any,
        lock,
        total_thread_count,
        total_input_idx,
        **extra_gen_kwargs: Any
    ) -> None:
        """Consume items from a shared inputs list and generate outputs concurrently, with optional rate limiting and progress logging.

        Args:
            shared_inputs (list): A process and thread-safe list supplying input data items.
            lock: process lock.
            total_thread_count (int): total thread count.
            total_input_idx (int): idx of inputs.
        Returns:
            None: This method runs until the queue is exhausted and then exits.
        """
        self.tqdm_pos = extra_gen_kwargs.get("process_id")
        max_out_len = extra_gen_kwargs.get("max_out_len", 1)
        concurrency = extra_gen_kwargs.get("concurrency")
        total_concurrency = extra_gen_kwargs.get("total_concurrency")
        pressure_time = PRESSURE_TIME
        connection_add_rate = CONNECTION_ADD_RATE
        if not isinstance(PRESSURE_TIME, (int, float)):
            raise TypeError("Type of PRESSURE_TIME is not int or float!")
        if not isinstance(CONNECTION_ADD_RATE, (int, float)):
            raise TypeError("Type of CONNECTION_ADD_RATE is not int or float!")

        if PRESSURE_TIME > PRESSURE_TIME_MAX:
            self.logger.warning(f"PRESSURE_TIME is larger than {PRESSURE_TIME_MAX}, will be set to {PRESSURE_TIME_MAX}")
            pressure_time = PRESSURE_TIME_MAX
        elif PRESSURE_TIME < PRESSURE_TIME_MIN:
            self.logger.warning(f"PRESSURE_TIME is smaller than {PRESSURE_TIME_MIN}, will be set to {PRESSURE_TIME_MIN}")
            pressure_time = PRESSURE_TIME_MIN
        if CONNECTION_ADD_RATE < CONNECTION_ADD_RATE_MIN:
            self.logger.warning(f"CONNECTION_ADD_RATE is smaller than {CONNECTION_ADD_RATE_MIN}, will be set to {CONNECTION_ADD_RATE_MIN}")
            connection_add_rate = CONNECTION_ADD_RATE_MIN

        thread_lock = threading.Lock()
        def generate_before_timeout(shared_inputs, total_input_idx, max_out_len):
            while(time.perf_counter() - self.start_time <= pressure_time):
                with lock:
                    with thread_lock:
                        cur_idx = total_input_idx.value % len(shared_inputs) #
                input_data = dict(data_id=total_input_idx.value, prompt=shared_inputs[cur_idx], gold="")
                total_input_idx.value += 1
                try:
                    _ = self._generate(input_data, max_out_len)
                except Exception as e:
                    self.logger.error(f"{e}")
                    continue

        self.token_bucket = None
        local_thread_count = 0

        draw_thread = threading.Thread(target=self.draw_plog_pressure, args=(self.tqdm_pos,), daemon=True)
        draw_thread.start()
        generate_threads = []
        self.start_time = time.perf_counter()
        try:
            while True:
                if total_thread_count.value >= total_concurrency or local_thread_count >= concurrency:
                    break
                if time.perf_counter() - self.start_time > pressure_time:
                    break
                with lock:
                    total_thread_count.value += 1
                local_thread_count += 1
                generate_thread = threading.Thread(target=generate_before_timeout, args=(shared_inputs, total_input_idx, max_out_len,))
                generate_thread.daemon = True
                generate_threads.append(generate_thread)
                generate_thread.start()
                time.sleep(1 / connection_add_rate)

        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user (Ctrl+C).")
        except Exception as e:
            self.logger.error(f"Infer task end because erro: {e}")
        finally:
            for thread in generate_threads:
                if thread.is_alive():
                    thread.join()
                    if thread.is_alive():
                        self.logger.warning(f"Thread {thread.ident} did not exit cleanly")
            self.task_finish = True
            self.tmp_result_queue.put(None)
            draw_thread.join()

    def flush(self):
        """Ensure simultaneous emptying of stdout and stderr when concurrent
        resources are available.

        When employing multiprocessing with standard I/O redirected to files,
        it is crucial to clear internal data for examination or prevent log
        loss in case of system failures."
        """
        if hasattr(self, 'tokens'):
            sys.stdout.flush()
            sys.stderr.flush()

    def acquire(self):
        """Acquire concurrent resources if exists.

        This behavior will fall back to wait with request_rate if there are
        no concurrent resources.
        """
        if hasattr(self, 'tokens'):
            self.tokens.acquire()
        else:
            self.wait()

    def release(self):
        """Release concurrent resources if acquired.

        This behavior will fall back to do nothing if there are no concurrent
        resources.
        """
        if hasattr(self, 'tokens'):
            self.tokens.release()

    @abstractmethod
    def get_ppl(self,
                inputs: List[PromptType],
                mask_length: Optional[List[int]] = None) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[PromptType]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """
        raise NotImplementedError(f'{self.__class__.__name__} does not support'
                                  ' ppl-based evaluation yet, try gen-based '
                                  'instead.')

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized string. Only English and Chinese
        characters are counted for now. Users are encouraged to override this
        method if more accurate length is needed.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """

        english_parts = re.findall(r'[A-Za-z0-9]+', prompt)
        chinese_parts = re.findall(r'[\u4e00-\u9FFF]+', prompt)

        # Count English words
        english_count = sum(len(part.split()) for part in english_parts)

        # Count Chinese words
        chinese_count = sum(len(part) for part in chinese_parts)

        return english_count + chinese_count

    def wait(self):
        """Wait till the next query can be sent.

        Applicable in both single-thread and multi-thread environments.
        """
        if not self.token_bucket:
            return
        return self.token_bucket.get_token()

    def to(self, device):
        pass


class APITemplateParser:
    """Intermidate prompt template parser, specifically for API models.

    Args:
        meta_template (Dict): The meta template for the model.
    """

    def __init__(self, meta_template: Optional[Dict] = None):
        self.meta_template = meta_template
        # Check meta template
        if meta_template:
            assert 'round' in meta_template, 'round is required in meta' \
                ' template'
            assert isinstance(meta_template['round'], list)
            keys_to_check = ['round']

            if 'reserved_roles' in meta_template:
                assert isinstance(meta_template['reserved_roles'], list)
                keys_to_check.append('reserved_roles')

            self.roles: Dict[str, dict] = dict()  # maps role name to config
            for meta_key in keys_to_check:
                for item in meta_template[meta_key]:
                    assert isinstance(item, (str, dict))
                    if isinstance(item, dict):
                        assert item['role'] not in self.roles, \
                            'role in meta prompt must be unique!'
                        self.roles[item['role']] = item.copy()

    def parse_template(self, prompt_template: PromptType,
                       mode: str) -> PromptType:
        """Parse the intermidate prompt template, and wrap it with meta
        template if applicable. When the meta template is set and the input is
        a PromptList, the return value will be a PromptList containing the full
        conversation history. Each item looks like:

        .. code-block:: python

            {'role': 'user', 'prompt': '...'}).

        Args:
            prompt_template (List[PromptType]): An intermidate prompt
                template (potentially before being wrapped by meta template).
            mode (str): Parsing mode. Choices are 'ppl' and 'gen'.

        Returns:
            List[PromptType]: The finalized prompt or a conversation.
        """
        assert isinstance(prompt_template, (str, list, PromptList, tuple))

        #mm data
        if is_mm_prompt(prompt_template):
            return prompt_template

        if not isinstance(prompt_template, (str, PromptList)):
            return [self.parse_template(p, mode=mode) for p in prompt_template]

        assert mode in ['ppl', 'gen']
        if isinstance(prompt_template, str):
            return prompt_template

        if self.meta_template:

            prompt = PromptList()
            # Whether to keep generating the prompt
            generate = True

            section_stack = []  # stores tuples: (section_name, start_idx)

            for i, item in enumerate(prompt_template):
                if not generate:
                    break
                if isinstance(item, str):
                    if item.strip():
                        # TODO: logger
                        warnings.warn('Non-empty string in prompt template '
                                      'will be ignored in API models.')
                elif isinstance(item, dict) and 'section' in item:
                    if item['pos'] == 'end':
                        section_name, start_idx = section_stack.pop(-1)
                        assert section_name == item['section']
                        if section_name in ['round', 'ice']:
                            dialogue = prompt_template[start_idx:i]
                            round_ranges = self._split_rounds(
                                dialogue, self.meta_template['round'])
                            # Consider inserting multiple round examples into
                            # template
                            for i in range(len(round_ranges) - 1):
                                start = round_ranges[i]
                                end = round_ranges[i + 1]
                                round_template = dialogue[start:end]
                                role_dict = self._update_role_dict(
                                    round_template)
                                api_prompts, generate = self._prompt2api(
                                    self.meta_template['round'],
                                    role_dict,
                                    # Start generating only when the mode is in
                                    # generation and the template reaches the
                                    # last round
                                    for_gen=mode == 'gen'
                                    and section_name == 'round'
                                    and i == len(round_ranges) - 2)
                                prompt += api_prompts
                    elif item['pos'] == 'begin':
                        assert item['section'] in [
                            'begin', 'round', 'end', 'ice'
                        ]
                        section_stack.append((item['section'], i + 1))
                    else:
                        raise ValueError(f'Invalid pos {item["pos"]}')
                elif section_stack[-1][0] in ['begin', 'end']:
                    role_dict = self._update_role_dict(item)
                    api_prompts, generate = self._prompt2api(
                        item, role_dict, for_gen=mode == 'gen')
                    prompt.append(api_prompts)

            # merge the consecutive prompts assigned to the same role
            new_prompt = PromptList([prompt[0]])
            last_role = prompt[0]['role']
            for item in prompt[1:]:
                if item['role'] == last_role:
                    new_prompt[-1]['prompt'] += '\n' + item['prompt']
                else:
                    last_role = item['role']
                    new_prompt.append(item)
            prompt = new_prompt

            if self.meta_template.get('begin', None):
                prompt.insert(0, self.meta_template['begin'])

        else:
            # in case the model does not have any meta template
            prompt = ''
            last_sep = ''
            for item in prompt_template:
                if isinstance(item, dict) and set(['section', 'pos']) == set(
                        item.keys()):
                    continue
                if isinstance(item, str):
                    if item:
                        prompt += last_sep + item
                elif item.get('prompt', ''):
                    prompt += last_sep + item.get('prompt', '')
                last_sep = '\n'
        return prompt

    def _update_role_dict(self, prompts: Union[List, str]) -> Dict[str, Dict]:
        """Update the default role dict with the given prompts."""
        role_dict = deepcopy(self.roles)
        if isinstance(prompts, str):
            return role_dict
        elif isinstance(prompts, dict):
            prompts = [prompts]
        for prompt in prompts:
            if isinstance(prompt, dict):
                role = prompt['role']
                if role not in self.roles:
                    role = prompt.get('fallback_role', None)
                    if not role:
                        print(f'{prompt} neither has an appropriate role nor '
                              'a fallback role.')
                role_dict[role].update(prompt)
        return role_dict

    def _split_rounds(
            self, prompt_template: List[Union[str, Dict]],
            single_round_template: List[Union[str, Dict]]) -> List[int]:
        """Split the prompt template into rounds, based on single round
        template.

        Return the index ranges of each round. Specifically,
        prompt_template[res[i]:res[i+1]] represents the i-th round in the
        template.
        """
        role_idxs = {
            role_cfg['role']: i
            for i, role_cfg in enumerate(single_round_template)
            if not isinstance(role_cfg, str)
        }
        last_role_idx = -1
        cutoff_idxs = [0]
        for idx, template in enumerate(prompt_template):
            if isinstance(template, str):
                continue
            role_idx = role_idxs.get(template['role'], None)
            if role_idx is None:
                try:
                    role_idx = role_idxs[template['fallback_role']]
                except KeyError:
                    raise KeyError(f'{template} neither has an appropriate '
                                   'role nor a fallback role.')
            if role_idx <= last_role_idx:
                cutoff_idxs.append(idx)
            last_role_idx = role_idx
        cutoff_idxs.append(len(prompt_template))
        return cutoff_idxs

    def _prompt2api(self,
                    prompts: Union[List, str],
                    role_dict: Dict[str, Dict],
                    for_gen: bool = False) -> Tuple[List, bool]:
        """Convert the prompts to a API-style prompts, given an updated
        role_dict.

        Args:
            prompts (Union[List, str]): The prompts to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[List, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        cont = True
        if isinstance(prompts, str):
            return prompts, cont
        elif isinstance(prompts, dict):
            api_role, cont = self._role2api_role(prompts, role_dict, for_gen)
            return api_role, cont

        res = []
        for prompt in prompts:
            if isinstance(prompt, str):
                raise TypeError('Mixing str without explicit role is not '
                                'allowed in API models!')
            else:
                api_role, cont = self._role2api_role(prompt, role_dict,
                                                     for_gen)
                if api_role:
                    res.append(api_role)
                if not cont:
                    break
        return res, cont

    def _role2api_role(self,
                       role_prompt: Dict,
                       role_dict: Dict[str, Dict],
                       for_gen: bool = False) -> Tuple[Dict, bool]:
        """Convert a role prompt to a string, given an updated role_dict.

        Args:
            role_prompt (Dict): The role prompt to be converted.
            role_dict (Dict[str, Dict]): The updated role dict.
            for_gen (bool): If True, the prompts will be converted for
                generation tasks. The conversion stops before the first
                role whose "generate" is set to True.

        Returns:
            Tuple[Dict, bool]: The converted string, and whether the follow-up
            conversion should be proceeded.
        """
        merged_prompt = role_dict.get(
            role_prompt['role'],
            role_dict.get(role_prompt.get('fallback_role')))
        # res_api_prompt = dict(type='', )
        if for_gen and merged_prompt.get('generate', False):
            return None, False
        res = {}
        res['role'] = merged_prompt['api_role']
        res['prompt'] = merged_prompt.get('begin', '')
        res['prompt'] += merged_prompt.get('prompt', '')
        res['prompt'] += merged_prompt.get('end', '')
        return res, True


class TokenBucket:
    """A token bucket for rate limiting.

    Args:
        sleep_intervals (List[float]): The sleep intervals of the token bucket.
    """

    def __init__(self, sleep_intervals: List[float], verbose=False, start_time=None):
        self.sleep_intervals = sleep_intervals
        self.total_intervals = len(self.sleep_intervals)
        self.current_index = 0
        self.start_time = start_time
        
        self._tokens = threading.Semaphore(0)
        self.started = False
        self._request_queue = Queue()
        self.logger = get_logger()
        self.verbose = verbose

    def _add_tokens(self):
        """Add tokens at precise precomputed times"""
        # Set start time on first call
        if self.start_time is None:
            self.start_time = time.perf_counter()
        
        while self.current_index < self.total_intervals:
            # Calculate target time for this token
            target_time = self.start_time + self.sleep_intervals[self.current_index]
            
            # Calculate wait time
            current_time = time.perf_counter()
            wait_time = max(0.0, target_time - current_time)
            
            # Sleep until target time
            if wait_time > 0:
                time.sleep(wait_time)
            
            # Release token
            self._tokens.release()
            
            # Move to next interval
            self.current_index += 1

    def get_token(self):
        """Get a token from the bucket."""
        if not self.started:
            self.started = True
            threading.Thread(target=self._add_tokens, daemon=True).start()
        self._tokens.acquire()
        if self.verbose:
            cur_time = time.perf_counter()
            while not self._request_queue.empty():
                if cur_time - self._request_queue.queue[0] > 60:
                    self._request_queue.get()
                else:
                    break
            self._request_queue.put(cur_time)
            self.logger.info(f'Current RPM {self._request_queue.qsize()}.')


def is_synthetic_string(input):
    pattern = r'^(A\s)+A[0-9]+$'
    if isinstance(input, dict):
        return bool(re.match(pattern, input.get('prompt')))
    else:
        return bool(re.match(pattern, input))


def handle_synthetic_input(func):
    def wrapper(self, input: str | dict, max_out_len: int) -> str:
        # 检查是否是synthetic输入
        if isinstance(input, dict):
            input_str = input.get('prompt')
            if not input_str:
                raise ValueError(f"Input dict:{input} has no prompt key")
        elif isinstance(input, (str, PromptList)):
            input_str = input
        else:
            raise TypeError(f"Excepted str or dict ,but got {type(input)}")
        if hasattr(self, "is_synthetic") and self.is_synthetic and is_synthetic_string(input_str):
            max_out_len = int(input_str.split('A')[-1])  # 提取 max_out_len
            input_str = input_str[:-len(input_str.split('A')[-1])]  # 去掉 max_out_len 部分
            if isinstance(input, dict):
                input['prompt'] = input_str
            else:
                input = input_str
        self.acquire()
        res = func(self, input, max_out_len)
        try:
            self.client.finish_count() # hf model has no client
        except AttributeError as e:
            pass
        if isinstance(input, dict) and  ((not hasattr(self, "do_performance")) or  (not self.do_performance)):
            tmp_res = {
                'data_id': input.get('data_id'),
                'origin_prompt': input.get('prompt'),
                'prediction': res,
            }
            if input.get('gold'):
                tmp_res['gold'] = input.get('gold')
            self.tmp_result_queue.put(tmp_res)
        return res
    return wrapper
