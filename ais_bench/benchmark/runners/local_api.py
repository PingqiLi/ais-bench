import logging
import os
import os.path as osp
import subprocess
import sys
import time
import traceback
from multiprocessing import Manager, Pool
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Tuple

import mmengine
from mmengine.config import ConfigDict
from tqdm import tqdm

from ais_bench.benchmark.registry import RUNNERS, TASKS
from ais_bench.benchmark.tasks import OpenICLInferTask, OpenICLPerfTask, OpenICLInferMergedTask
from ais_bench.benchmark.tasks.base import BaseTask
from ais_bench.benchmark.utils import (build_dataset_from_cfg, build_dataset_from_cfg_with_model_path,
                                       build_model_from_cfg, get_infer_output_path,
                                       get_logger, task_abbr_from_cfg)
from ais_bench.benchmark.utils.types import _check_type

from .base import BaseRunner

pbar_counter = 0

def monkey_run_perf(self):
    self.logger.info(f"Task {task_abbr_from_cfg(self.cfg)}")
    for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
        self.max_out_len = model_cfg.get("max_out_len", None)
        self.batch_size = model_cfg.get("batch_size", None)
        self.min_out_len = model_cfg.get("min_out_len", None)

        self.model = build_model_from_cfg(model_cfg)
        self.set_performance_api()

        for dataset_cfg in dataset_cfgs:
            self.model_cfg = model_cfg
            self.dataset_cfg = dataset_cfg
            self.infer_cfg = self.dataset_cfg["infer_cfg"]
            if self.dataset_cfg.get('type', None) in ["ais_bench.benchmark.datasets.SyntheticDataset",
                                                      "ais_bench.benchmark.datasets.ShareGPTDataset"]:
                self.dataset = build_dataset_from_cfg_with_model_path(self.dataset_cfg, self.model_cfg)
            else:
                self.dataset = build_dataset_from_cfg(self.dataset_cfg)
            self.build_inference()
            self.sub_cfg = {
                "models": [self.model_cfg],
                "datasets": [[self.dataset_cfg]],
            }
            out_path = get_infer_output_path(
                self.model_cfg,
                self.dataset_cfg,
                osp.join(self.work_dir, "performances"),
            )
            if osp.exists(out_path):
                continue
            entry, golds = self.get_data_list()
            self.entry.extend(entry)
            self.golds.extend(golds)
        self.do_performance()

def monkey_run_merged(self):
    self.logger.info(f"Task {task_abbr_from_cfg(self.cfg)}")
    for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
        self.max_out_len = model_cfg.get("max_out_len", None)
        self.batch_size = model_cfg.get("batch_size", None)
        self.min_out_len = model_cfg.get("min_out_len", None)

        self.model = build_model_from_cfg(model_cfg)
        
        num_return_sequences = getattr(model_cfg, 'generation_kwargs', {}).pop('num_return_sequences', 1)
        _check_type(num_return_sequences, int)
        assert num_return_sequences > 0, f"num_return_sequences expected a positive integer, but got {num_return_sequences}"

        for dataset_cfg in dataset_cfgs:
            self.model_cfg = model_cfg
            self.dataset_cfg = dataset_cfg
            self.infer_cfg = self.dataset_cfg["infer_cfg"]
            
            if 'n' not in self.dataset_cfg:
                    self.dataset_cfg['n'] = num_return_sequences
            _check_type(self.dataset_cfg['n'], int)
            assert self.dataset_cfg['n'] > 0, f"n expected a positive integer, but got {self.dataset_cfg['n']}"
            
            self.dataset = build_dataset_from_cfg(self.dataset_cfg)
            self.build_inference()
            self.sub_cfg = {
                "models": [self.model_cfg],
                "datasets": [[self.dataset_cfg]],
            }
            entry, golds = self.get_data_list()
            self.entry.extend(entry)
            self.golds.extend(golds)
        self.do_inference()

def monkey_run(self):
    """Hack for infer task run, add tokens for multiprocess."""
    self.logger.info(f'Task {task_abbr_from_cfg(self.cfg)}')
    for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
        self.max_out_len = model_cfg.get('max_out_len', None)
        self.min_out_len = model_cfg.get('min_out_len', None)
        self.batch_size = model_cfg.get('batch_size', 1)
        self.model = build_model_from_cfg(model_cfg)
        # add global tokens for concurrents
        # assert self.model.is_api, 'Only API model is supported.'
        
        num_return_sequences = getattr(model_cfg, 'generation_kwargs', {}).pop('num_return_sequences', 1)
        _check_type(num_return_sequences, int)
        assert num_return_sequences > 0, f"num_return_sequences expected a positive integer, but got {num_return_sequences}"

        for dataset_cfg in dataset_cfgs:
            self.model_cfg = model_cfg
            self.dataset_cfg = dataset_cfg
            self.infer_cfg = self.dataset_cfg['infer_cfg']

            if 'n' not in self.dataset_cfg:
                    self.dataset_cfg['n'] = num_return_sequences
            _check_type(self.dataset_cfg['n'], int)
            assert self.dataset_cfg['n'] > 0, f"n expected a positive integer, but got {self.dataset_cfg['n']}"

            self.dataset = build_dataset_from_cfg(self.dataset_cfg)
            self.sub_cfg = {
                'models': [self.model_cfg],
                'datasets': [[self.dataset_cfg]],
            }
            self._inference()


old_stdout = sys.stdout
old_stderr = sys.stderr


def redirect_std_to_file(filename: str):
    """Redirect stdout and stderr, also change logger stream handler."""
    f = open(filename, 'w', encoding='utf-8')
    sys.stdout = f
    sys.stderr = f
    # change logger stream handler as well
    logger = get_logger()
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout
    # special treat for icl_gen_inferencer logger
    gen_logger = logging.getLogger(
        'ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer')
    for h in gen_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout


def reset_std():
    """Reset stdout and stderr, also change logger stream handler."""
    sys.stdout.close()
    sys.stdout = old_stdout
    sys.stderr = old_stderr
    # change logger stream handler as well
    logger = get_logger()
    for h in logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout
    # special treat for icl_gen_inferencer logger
    gen_logger = logging.getLogger(
        'ais_bench.benchmark.openicl.icl_inferencer.icl_gen_inferencer')
    for h in gen_logger.handlers:
        if isinstance(h, logging.StreamHandler):
            h.stream = sys.stdout


def launch(task: BaseTask):
    """Launch a single task.

    Args:
        task (BaseTask): Task to launch.
        tokens (SyncManager.Semaphore): Multiprocessing semaphore
            for every subprocess to follow.

    Returns:
        tuple[str, int]: Task name and exit code.
    """

    task_name = task.name
    returncode = 0
    logger = get_logger()

    try:
        # get log file and redirect stdout and stderr
        out_path = task.get_log_path(file_extension='out')
        mmengine.mkdir_or_exist(osp.split(out_path)[0])
        redirect_std_to_file(out_path)

        # start infer with monkey_run
        start_time = time.perf_counter()
        if task.name_prefix == 'OpenICLPerf':
            inferencer = OpenICLPerfTask(task.cfg)
            origin_run = inferencer.run
            inferencer.run = monkey_run_perf
        elif task.name_prefix == 'OpenICLInferMerged':
            inferencer = OpenICLInferMergedTask(task.cfg)
            origin_run = inferencer.run
            inferencer.run = monkey_run_merged
        else:
            inferencer = OpenICLInferTask(task.cfg)
            origin_run = inferencer.run
            inferencer.run = monkey_run
        inferencer.run(inferencer)
        inferencer.run = origin_run
        end_time = time.perf_counter()
        logger.info(f'time elapsed: {end_time - start_time:.2f}s')
    except Exception:
        # print trace back in target file
        traceback.print_exc()
        # reset stdout and stderr
        reset_std()
        logger.error(f'task {task_name} fail, see\n{out_path}')
        returncode = 1
    else:
        # reset stdout and stderr
        reset_std()
    return task_name, returncode


def submit(task, type):
    """Helper for launch the task."""
    task = TASKS.build(dict(cfg=task, type=type))
    tqdm.write(f'Launch {task.name} on CPU ')

    res = launch(task)
    return res


@RUNNERS.register_module()
class LocalAPIRunner(BaseRunner):
    """Local API Runner. Start tasks by local python.

    The query per second cannot guarantee the number of concurrents, therefore
    Supported concurrent users with multiple tasks. Applied for those apis
    which has a restriction on concurrent numbers.

    Args:
        task (ConfigDict): Task type config.
        max_num_workers (int): Max number of workers to run in parallel.
            Defaults to 16.
        debug (bool): Whether to run in debug mode.
        lark_bot_url (str): Lark bot url.
    """

    def __init__(self,
                 task: ConfigDict,
                 max_num_workers: int = 16,
                 num_prompts: int = None,
                 debug: bool = False,
                 disable_cb: bool = False,
                 lark_bot_url: str = None):
        super().__init__(task=task, debug=debug, lark_bot_url=lark_bot_url)
        self.max_num_workers = max_num_workers
        self.num_prompts = num_prompts
        self.disable_cb = disable_cb
        get_logger().debug(f"task type is {task['type']}")
        assert task['type'] in [
            'OpenICLInferTask',
            'ais_bench.benchmark.tasks.openicl_infer.OpenICLInferTask',
            'ais_bench.benchmark.tasks.OpenICLInferTask',
            'OpenICLInferMergedTask',
            'ais_bench.benchmark.tasks.openicl_infer_merged.OpenICLInferMergedTask',
            'ais_bench.benchmark.tasks.OpenICLInferMergedTask',
            'OpenICLPerfTask',
            'ais_bench.benchmark.tasks.openicl_perf.OpenICLPerfTask',
            'ais_bench.benchmark.tasks.OpenICLPerfTask',
        ], 'Only supported for api infer task.'

    def launch(self, tasks: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Launch multiple tasks.

        Args:
            tasks (list[dict]): A list of task configs, usually generated by
                Partitioner.

        Returns:
            list[tuple[str, int]]: A list of (task name, exit code).
        """
        status = []
        if self.debug:
            # fall back to LocalRunner debug mode
            for task in tqdm(tasks, desc="Running tasks"):
                task['num_prompts'] = self.num_prompts
                task = TASKS.build(dict(cfg=task, type=self.task_cfg['type']))
                task_name = task.name
                # get cmd
                mmengine.mkdir_or_exist('tmp/')
                param_file = f'tmp/{os.getpid()}_params.py'
                try:
                    task.cfg.dump(param_file)
                    cmd = task.get_command(cfg_path=param_file,
                                        template='{task_cmd}')
                    # run in subprocess if starts with torchrun etc.
                    if cmd.startswith('python'):
                        task.run()
                    else:
                        proc = subprocess.Popen(cmd, shell=True, text=True)
                    try:
                        proc.wait()
                    except KeyboardInterrupt:
                        get_logger().warning(f"Subprocess of task:{task_name} interrupted by user!")
                        proc.wait()  # 确保子进程执行完毕

                finally:
                    os.remove(param_file)
                status.append((task_name, 0))
        elif not self.disable_cb:
            get_logger().info('Continuous batch enable! All the logs and processes for each task'
                                ' should be checked in each infer/.out file.')
            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for task in tasks:
                    task['num_prompts'] = self.num_prompts
                    res = submit(task, self.task_cfg['type'])
                    status.append(res)
                    pbar.update(1)
        else:
            pbar = tqdm(total=len(tasks))

            get_logger().info('All the logs and processes for each task'
                              ' should be checked in each infer/.out file.')
            with Manager() as manager:
                # pbar update has visualization issue when direct
                # update pbar in callback, need an extra counter
                pbar_counter = manager.Value('i', 0)
                status = []

                def update(args):
                    """Update pbar counter when callback."""
                    pbar_counter.value += 1
                    status.append(args)

                with Pool(processes=self.max_num_workers) as pool:
                    for task in tasks:
                        task['num_prompts'] = self.num_prompts
                        pool.apply_async(submit,
                                         (task, self.task_cfg['type']),
                                         callback=update)
                    pool.close()

                    # update progress bar
                    while True:
                        cur_count = pbar_counter.value
                        if cur_count > pbar.n:
                            pbar.update(cur_count - pbar.n)
                        # break when all the task finished
                        if cur_count >= pbar.total:
                            pbar.close()
                            break
                        # sleep to lower the usage
                        time.sleep(1)

                    pool.join()
        return status
