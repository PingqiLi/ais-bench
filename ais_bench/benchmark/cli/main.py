# flake8: noqa
# yapf: disable
import argparse
import copy
import getpass
import os
import os.path as osp
from datetime import datetime

from mmengine.config import Config, DictAction

from ais_bench.benchmark.registry import PARTITIONERS, RUNNERS, build_from_cfg
from ais_bench.benchmark.summarizers import DefaultSummarizer, DefaultPerfSummarizer
from ais_bench.benchmark.calculators import DefaultPerfMetricCalculator
from ais_bench.benchmark.utils import LarkReporter, get_logger
from ais_bench.benchmark.utils.tokenizer import BenchmarkTokenizer
from ais_bench.benchmark.utils.run import (fill_infer_cfg, fill_eval_cfg, get_config_from_arg, fill_perf_cfg,
    fill_merged_infer_cfg, fill_merged_eval_cfg)

def get_current_time_str():
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def parse_args():
    parser = argparse.ArgumentParser(description='Run an evaluation task')
    parser.add_argument('config', nargs='?', help='Benchmark config file path')

    parser.add_argument('--models', nargs='+', help='', default=None)
    parser.add_argument('--datasets', nargs='+', help='', default=None)
    parser.add_argument('--summarizer', help='', default=None)
    # add general args
    parser.add_argument('--debug',
                        help='Debug mode, in which scheduler will run tasks '
                        'in the single process, and output will not be '
                        'redirected to files',
                        action='store_true',
                        default=False)
    parser.add_argument('--dry-run',
                        help='Dry run mode, in which the scheduler will not '
                        'actually run the tasks, but only print the commands '
                        'to run',
                        action='store_true',
                        default=False)
    parser.add_argument('-m',
                        '--mode',
                        help='Running mode. Choose "perf" for performance evaluation, "infer" to run inference only, '
                        '"eval" to evaluate existing inference results, or "viz" to visualize the results. '
                        'The default mode is "all", which runs all steps.',
                        choices=['all', 'infer', 'eval', 'viz', 'perf', 'perf_viz'],
                        default='all',
                        type=str)
    parser.add_argument('-r',
                        '--reuse',
                        nargs='?',
                        type=str,
                        const='latest',
                        help='Reuse previous outputs & results, and run any '
                        'missing jobs presented in the config. If its '
                        'argument is not specified, the latest results in '
                        'the work_dir will be reused. The argument should '
                        'also be a specific timestamp, e.g. 20230516_144254')
    parser.add_argument('-w',
                        '--work-dir',
                        help='Work path, all the outputs will be '
                        'saved in this path, including the predictions, '
                        'the evaluation results, the summary results, etc.'
                        'If not specified, the work_dir will be set to '
                        'outputs/default.',
                        default=None,
                        type=str)
    parser.add_argument(
        '--config-dir',
        default='configs',
        help='Use the custom config directory instead of config/ to '
        'search the configs for datasets, models and summarizers',
        type=str)
    parser.add_argument('--max-num-workers',
                        help='Max number of workers to run in parallel. '
                        'Will be overrideen by the "max_num_workers" argument '
                        'in the config.',
                        type=int,
                        default=1)
    parser.add_argument('--num-prompts',
                        help='Num Prompts, Specify the number of prompts to evaluate. '
                        'If not provided, all prompts will be evaluated. ',
                        type=int,
                        default=None)

    # evaluatation add
    parser.add_argument(
        '--max-workers-per-gpu',
        help='Max task to run in parallel on one GPU. '
        'It will only be used in the local runner.',
        type=int,
        default=1
    )
    parser.add_argument(
        '--dump-eval-details',
        help='Whether to dump the evaluation details, including the '
        'correctness of each sample, bpb, etc.',
        action='store_true',
    )
    parser.add_argument(
        '--dump-extract-rate',
        help='Whether to dump the extract rate of evaluation (samples per sec)',
        action='store_true',
    )

    parser.add_argument(
        '--merge-ds',
        help='Whether to merge dataset with multi files(mmlu, ceval)',
        action='store_true',
    )

    parser.add_argument(
        '--pressure',
        help='Whether to enable pressure test in perf mode (only attr service)',
        action='store_true',
    )

    parser.add_argument(
        '--disable-cb',
        help='Whether to disable infer with continous batch mode',
        action='store_true',
    )

    # set custom dataset args
    custom_dataset_parser = parser.add_argument_group('custom_dataset_args')
    parse_custom_dataset_args(custom_dataset_parser)

    args = parser.parse_args()
    return args

def parse_custom_dataset_args(custom_dataset_parser):
    """These args are all for the quick construction of custom datasets."""
    custom_dataset_parser.add_argument('--custom-dataset-path', type=str)
    custom_dataset_parser.add_argument('--custom-dataset-data-type',
                                       type=str,
                                       choices=['mcq', 'qa'])
    custom_dataset_parser.add_argument('--custom-dataset-infer-method',
                                       type=str,
                                       choices=['gen'])

def main():
    args = parse_args()

    if args.dry_run:
        args.debug = True
    # initialize logger
    logger = get_logger(log_level='DEBUG' if args.debug else 'INFO')

    cfg = get_config_from_arg(args)
    if args.work_dir is not None:
        cfg['work_dir'] = args.work_dir
    else:
        cfg.setdefault('work_dir', os.path.join('outputs', 'default'))

    # cfg_time_str defaults to the current time
    cfg_time_str = dir_time_str = get_current_time_str()
    if args.reuse:
        if args.reuse == 'latest':
            if not os.path.exists(cfg.work_dir) or not os.listdir(
                    cfg.work_dir):
                logger.warning('No previous experiment results found to reuse.')
            else:
                dirs = os.listdir(cfg.work_dir)
                dir_time_str = sorted(dirs)[-1]
        else:
            dir_time_str = args.reuse
        logger.info(f'Reusing experiements from {dir_time_str}')

    # update "actual" work_dir
    cfg['work_dir'] = osp.join(cfg.work_dir, dir_time_str)
    current_workdir = cfg['work_dir']
    logger.info(f'Current exp folder: {current_workdir}')

    os.makedirs(osp.join(cfg.work_dir, 'configs'), exist_ok=True)

    # dump config
    output_config_path = osp.join(cfg.work_dir, 'configs',
                                  f'{cfg_time_str}_{os.getpid()}.py')
    cfg.dump(output_config_path)
    # eval nums set
    if (args.num_prompts and args.num_prompts < 0) or args.num_prompts == 0:
        raise ValueError("Num Prompts must be a positive integer greater than 0.")
    cfg['num_prompts'] = args.num_prompts
    # Config is intentally reloaded here to avoid initialized
    # types cannot be serialized
    cfg = Config.fromfile(output_config_path, format_python_code=False)

    if args.mode == 'perf':
        fill_perf_cfg(cfg, args)
        cfg.infer.partitioner['out_dir'] = osp.join(cfg['work_dir'],
                                                    'performances')
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        logger.info("Starting performance evaluation tasks...")
        tasks = partitioner(cfg)
        runner = RUNNERS.build(cfg.infer.runner)
        runner(tasks)
        logger.info("Performance evaluation tasks completed.")

    if cfg.get('infer', None) is None:
        if args.merge_ds:
            fill_merged_infer_cfg(cfg, args)
        else:
            fill_infer_cfg(cfg, args)

    if args.mode in ['all', 'infer']:
        # "infer" in config, we will provide a default configuration
        # for infer

        if args.debug:
            cfg.infer.runner.debug = True
        cfg.infer.partitioner['out_dir'] = osp.join(cfg['work_dir'],
                                                    'predictions/')
        cfg.dump(output_config_path)
        partitioner = PARTITIONERS.build(cfg.infer.partitioner)
        logger.info("Starting inference tasks...")
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.infer.runner)
        # Add extra attack config if exists
        if hasattr(cfg, 'attack'):
            for task in tasks:
                cfg.attack.dataset = task.datasets[0][0].abbr
                task.attack = cfg.attack
        runner(tasks)
        logger.info("Inference tasks completed.")

    if cfg.get('eval', None) is None:
        if args.merge_ds:
            fill_merged_eval_cfg(cfg, args)
        else:
            fill_eval_cfg(cfg, args)

    if args.mode in ['all', 'eval']:
        # "eval" in config, we will provide a default configuration
        # for eval

        if args.dump_eval_details:
            cfg.eval.runner.task.dump_details = True
        if args.dump_extract_rate:
            cfg.eval.runner.task.cal_extract_rate = True
        if args.debug:
            cfg.eval.runner.debug = True
        cfg.eval.partitioner['out_dir'] = osp.join(cfg['work_dir'], 'results/')
        cfg.dump(output_config_path)
        partitioner = PARTITIONERS.build(cfg.eval.partitioner)
        logger.info("Starting evaluation tasks...")
        tasks = partitioner(cfg)
        if args.dry_run:
            return
        runner = RUNNERS.build(cfg.eval.runner)

        # For meta-review-judge in subjective evaluation
        if isinstance(tasks, list) and len(tasks) != 0 and isinstance(tasks[0], list):
            for task_part in tasks:
                runner(task_part)
        else:
            runner(tasks)
        logger.info("Evaluation tasks completed.")

    # visualize accuracy results
    if args.mode in ['all', 'eval', 'viz']:
        logger.info("Summarizing evaluation results...")
        summarizer_cfg = cfg.get('summarizer', {})

        # For subjective summarizer
        if summarizer_cfg.get('function', None):
            main_summarizer_cfg = copy.deepcopy(summarizer_cfg)
            grouped_datasets = {}
            for dataset in cfg.datasets:
                prefix = dataset['abbr'].split('_')[0]
                if prefix not in grouped_datasets:
                    grouped_datasets[prefix] = []
                grouped_datasets[prefix].append(dataset)
            dataset_score_container = []
            for dataset in grouped_datasets.values():
                temp_cfg = copy.deepcopy(cfg)
                temp_cfg.datasets = dataset
                summarizer_cfg = dict(type=dataset[0]['summarizer']['type'], config=temp_cfg)
                summarizer = build_from_cfg(summarizer_cfg)
                dataset_score = summarizer.summarize(time_str=cfg_time_str)
                if dataset_score:
                    dataset_score_container.append(dataset_score)
            main_summarizer_cfg['config'] = cfg
            main_summarizer = build_from_cfg(main_summarizer_cfg)
            main_summarizer.summarize(time_str=cfg_time_str, subjective_scores=dataset_score_container)
        else:
            if not summarizer_cfg or summarizer_cfg.get('type', None) is None:
                summarizer_cfg['type'] = DefaultSummarizer
            summarizer_cfg['config'] = cfg
            summarizer = build_from_cfg(summarizer_cfg)
            summarizer.summarize(time_str=cfg_time_str)

    # visualize performance results
    if args.mode in ['perf', 'perf_viz']:
        summarizer_cfg = cfg.get('summarizer', {})
        if not summarizer_cfg or summarizer_cfg.get('type', None) is None:
            summarizer_cfg['type'] = DefaultPerfSummarizer
        summarizer_cfg['config'] = cfg
        if summarizer_cfg.get('calculator') is None:
            summarizer_cfg['calculator'] = dict(type=DefaultPerfMetricCalculator)
        summarizer_cfg.pop('dataset_abbrs', None)
        summarizer_cfg.pop('summary_groups', None)
        summarizer_cfg.pop('prompt_db', None)
        summarizer = build_from_cfg(summarizer_cfg)
        logger.info("Summarizing performance results...")
        summarizer.summarize()


if __name__ == '__main__':
    main()
