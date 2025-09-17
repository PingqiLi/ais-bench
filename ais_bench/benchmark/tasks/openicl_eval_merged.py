import argparse
import copy
import fnmatch
import math
import os
import os.path as osp
import statistics
import sys
import time
from collections import Counter
from inspect import signature
from typing import List
from datasets import concatenate_datasets

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.utils import mkdir_or_exist

from ais_bench.benchmark.registry import (ICL_EVALUATORS, MODELS, TASKS,
                                  TEXT_POSTPROCESSORS)
from ais_bench.benchmark.tasks.base import BaseTask, extract_role_pred
from ais_bench.benchmark.utils import (build_dataset_from_cfg, dataset_abbr_from_cfg,
                               get_infer_merged_output_path, get_logger,
                               task_abbr_from_cfg)
from ais_bench.benchmark.tasks.openicl_eval import OpenICLEvalTask
from ais_bench.benchmark.utils.types import _check_type


@TASKS.register_module()
class OpenICLEvalMergedTask(OpenICLEvalTask):
    """OpenICL Evaluation Task.

    This task is used to evaluate the metric between predictions and
    references.
    """

    name_prefix = 'OpenICLEvalMerged'
    log_subdir = 'logs/eval'
    output_subdir = 'results'

    def __init__(self, cfg: ConfigDict):
        super().__init__(cfg)

    def get_command(self, cfg_path, template):
        sys.path.append(os.getcwd())
        script_path = __file__
        python = sys.executable
        command = f'{python} {script_path} {cfg_path}'
        return template.format(task_cmd=command)

    def run(self):
        for model_cfg, dataset_cfgs in zip(self.model_cfgs, self.dataset_cfgs):
            self.merge_datasets_cfgs = {}
            num_return_sequences = getattr(model_cfg, 'generation_kwargs', {}).get('num_return_sequences', 1)

            for dataset_cfg in dataset_cfgs:
                merged_ds_abbr = dataset_cfg.get('type').split('.')[-1].lower()
                if self.merge_datasets_cfgs.get(merged_ds_abbr) is None:
                    self.merge_datasets_cfgs[merged_ds_abbr] = []
                    
                k = dataset_cfg.get('k', num_return_sequences)
                n = dataset_cfg.get('n', num_return_sequences)
                _check_type(k, int)
                assert k > 0, f"k expected a positive integer, but got {k}"
                _check_type(n, int)
                assert n > 0, f"n expected a positive integer, but got {n}"
                dataset_cfg["k"] = k
                dataset_cfg["n"] = n

                self.merge_datasets_cfgs[merged_ds_abbr].append(dataset_cfg)

            if not dataset_cfgs:
                return
            dataset_cfgs = dataset_cfgs[0]
            self.model_cfg = model_cfg
            self.dataset_cfg = dataset_cfg

            # Load Dataset
            self.eval_cfg = self.dataset_cfg.get('eval_cfg')
            self.output_column = dataset_cfg['reader_cfg']['output_column']

            # overwrite postprocessor if the model has specified one
            ds_abbr = dataset_abbr_from_cfg(self.dataset_cfg)
            model_postprocessors = self.model_cfg.get(
                'pred_postprocessor', {})
            for pattern in model_postprocessors.keys():
                if fnmatch.fnmatch(ds_abbr, pattern):
                    self.eval_cfg[
                        'pred_postprocessor'] = model_postprocessors[
                            pattern]  # noqa
                    break

            out_path = get_infer_merged_output_path(
                self.model_cfg, self.dataset_cfg,
                osp.join(self.work_dir, 'results'))
            if osp.exists(out_path):
                self.logger.warning(f'Output file {out_path} already exists and will be overwritten.')
            self._score()

    def _score(self):
        merged_ds_abbr = self.dataset_cfg.get('type').split('.')[-1].lower()
        test_set_list = []
        ds_cfgs = self.merge_datasets_cfgs.get(merged_ds_abbr)
        k = ds_cfgs[0].get('k', 1) if ds_cfgs else 1
        n = ds_cfgs[0].get('n', 1) if ds_cfgs else 1
        test_set_list = [build_dataset_from_cfg(ds_cfg).test for ds_cfg in ds_cfgs]

        # Postprocess dataset if necessary
        if 'dataset_postprocessor' in self.eval_cfg:
            proc = self.eval_cfg['dataset_postprocessor']['type']
            if isinstance(proc, str):
                proc = TEXT_POSTPROCESSORS.get(proc)

            def postprocess(sample):
                s = sample[self.output_column]
                sample[self.output_column] = proc(s)
                return sample
            test_set_list = [t_set.map(postprocess) for t_set in test_set_list]
        test_set = concatenate_datasets(test_set_list)

        # Load predictions
        filename = get_infer_merged_output_path(
            self.model_cfg, self.dataset_cfg,
            osp.join(self.work_dir, 'predictions'))
        # in case the prediction is partial
        root, ext = osp.splitext(filename)
        partial_filename = root + '_0' + ext

        # Get sc_size if use Self-Consistency
        sc_size = self.eval_cfg.get('sc_size')

        if not osp.exists(osp.realpath(filename)) and not osp.exists(
                osp.realpath(partial_filename)):
            result = {'error': 'No predictions found.'}
        else:
            if osp.exists(osp.realpath(filename)):
                preds = mmengine.load(filename)
                preds = [preds[str(i)] for i in range(len(preds))]
            else:
                filename = partial_filename
                preds = []
                i = 1
                while osp.exists(osp.realpath(filename)):
                    sub_preds = mmengine.load(filename)
                    preds.extend(
                        [sub_preds[str(i)] for i in range(len(sub_preds))])
                    filename = root + f'_{i}' + ext
                    i += 1
            pred_dicts = copy.deepcopy(preds)
            preds = {k: [pred.get(k) for pred in preds] for k in preds[0]}

            pred_strs = preds.pop('prediction', None)
            pred_list_flag = pred_strs is not None and isinstance(
                pred_strs[0], list)
            if ('pred_role' in self.eval_cfg
                    and 'meta_template' in self.model_cfg
                    and not MODELS.get(self.model_cfg['type']).is_api):
                # Create a prompt template for role config parsing
                from ais_bench.benchmark.models.base import LMTemplateParser
                parser = LMTemplateParser(self.model_cfg['meta_template'])
                role = parser.roles[self.eval_cfg['pred_role']]
                if sc_size is not None:
                    assert pred_list_flag, (
                        'The prediction for Self-Consistency'
                        'must be list.')
                if pred_list_flag:
                    pred_strs = [[
                        extract_role_pred(_pred, role.get('begin', None),
                                          role.get('end', None))
                        for _pred in pred
                    ] for pred in pred_strs]
                else:
                    pred_strs = [
                        extract_role_pred(pred, role.get('begin', None),
                                          role.get('end', None))
                        for pred in pred_strs
                    ]
            # Postprocess predictions in model cfg
            if 'pred_postprocessor' in self.model_cfg:
                kwargs = copy.deepcopy(self.model_cfg['pred_postprocessor'])
                proc = kwargs.pop('type')
                if isinstance(proc, str):
                    proc = TEXT_POSTPROCESSORS.get(proc)
                if pred_list_flag:
                    pred_strs = [[proc(s, **kwargs) for s in preds]
                                for preds in pred_strs]
                else:
                    pred_strs = [proc(s, **kwargs) for s in pred_strs]
                    
            # Postprocess predictions in eval cfg
            if 'pred_postprocessor' in self.eval_cfg:
                kwargs = self.eval_cfg['pred_postprocessor']
                proc = kwargs.pop('type')
                if isinstance(proc, str):
                    proc = TEXT_POSTPROCESSORS.get(proc)
                if pred_list_flag:
                    pred_strs = [[proc(s, **kwargs) for s in preds]
                                 for preds in pred_strs]
                else:
                    pred_strs = [proc(s, **kwargs) for s in pred_strs]

            model_pred_strs = []
            if 'model_postprocessor' in self.eval_cfg:
                references = (test_set[self.output_column]
                              if self.output_column else None)
                model_pred_dicts = copy.deepcopy(pred_dicts)
                for i, pred_dict in enumerate(model_pred_dicts):
                    pred_dict['reference'] = [references[i]]
                self.logger.info('Postprocessing model predictions...')
                kwargs = self.eval_cfg['model_postprocessor']
                proc = kwargs.pop('type')
                if isinstance(proc, str):
                    proc = TEXT_POSTPROCESSORS.get(proc)
                if pred_list_flag:
                    model_pred_strs = [[
                        proc(model_pred_dict, **kwargs)
                        for model_pred_dict in model_pred_dicts
                    ]]
                else:
                    model_pred_strs = proc(model_pred_dicts, **kwargs)

            # Get majority voting predictions if use self-consistency
            if sc_size is not None:
                pred_strs = [
                    Counter(s).most_common(1)[0][0] for s in pred_strs
                ]

            icl_evaluator = ICL_EVALUATORS.build(self.eval_cfg['evaluator'])
            # need results dir to save other files
            out_path = get_infer_merged_output_path(
                self.model_cfg, self.dataset_cfg,
                osp.join(self.work_dir, 'results'))
            icl_evaluator._out_dir = osp.splitext(out_path)[
                0]  # strip extension

            preds['predictions'] = pred_strs
            preds['references'] = (test_set[self.output_column]
                                   if self.output_column else None)
            preds['test_set'] = test_set
            if 'origin_prompt' not in preds:
                try:
                    preds['origin_prompt'] = [
                        None for _ in range(len(pred_strs))
                    ]
                except TypeError:
                    preds['origin_prompt'] = None
            preds = {
                k: preds[k]
                for k in signature(icl_evaluator.score).parameters
            }
            result = icl_evaluator.evaluate(k, n, copy.deepcopy(test_set), **preds)

            # Get model postprocess result
            model_details = None
            model_result = None
            if 'model_postprocessor' in self.eval_cfg:
                model_preds = copy.deepcopy(preds)
                model_preds['predictions'] = model_pred_strs
                model_result = icl_evaluator.score(**model_preds)
                for key in model_result:
                    if key == 'details':
                        model_details = model_result[key]
                        continue
                    new_key = 'model_postprocess_' + key
                    result[new_key] = model_result[key]

            if self.dump_details:
                details = result.get('details', None)
                try:
                    result['details'] = self.format_details(
                        pred_strs, model_pred_strs,
                        test_set[self.output_column], details, model_details,
                        pred_dicts)
                    result['type'] = result['details'].pop('type', None)
                    if self.cal_extract_rate:
                        # Calculate the extraction success rate for prediction
                        result['extract_rate'] = self.extract_rate(result)

                    if 'PPL' in str(
                            self.dataset_cfg.infer_cfg.inferencer.type):
                        result['correct_bpb'], result['incorrect_bpb'] = \
                            self.calculate_bpb(pred_dicts)
                except Exception as e:
                    self.logger.warning(f'Skip dumping details due to: {e}.')
            else:
                result.pop('details', None)

        if 'error' in result:
            self.logger.error(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result["error"]}')
            return
        elif model_result is None:
            result_wo_details = {
                i: result[i]
                for i in result if i != 'details'
            }
            self.logger.info(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result_wo_details}')
        else:
            result_wo_details = {
                i: result[i]
                for i in result if i != 'details'
            }
            model_result_wo_details = {
                i: model_result[i]
                for i in model_result if i != 'details'
            }
            self.logger.info(
                f'Task {task_abbr_from_cfg(self.cfg)}: {result_wo_details}')
            self.logger.info(
                'Model Postprocess Task: ' +
                f'{task_abbr_from_cfg(self.cfg)}:{model_result_wo_details}')

        # Save result
        out_path = get_infer_merged_output_path(self.model_cfg, self.dataset_cfg,
                                         osp.join(self.work_dir, 'results'))
        mkdir_or_exist(osp.split(out_path)[0])
        mmengine.dump(result, out_path, ensure_ascii=False, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description='Score Calculator')
    parser.add_argument('config', help='Config file path')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    cfg = Config.fromfile(args.config)
    start_time = time.perf_counter()
    inferencer = OpenICLEvalMergedTask(cfg)
    inferencer.run()
    end_time = time.perf_counter()
    get_logger().info(f'time elapsed: {end_time - start_time:.2f}s')
