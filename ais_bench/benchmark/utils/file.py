import fnmatch
import tabulate
import os
from typing import List, Tuple, Union
from ais_bench.benchmark.utils.logging import get_logger

logger = get_logger()


def match_files(path: str,
                pattern: Union[str, List],
                fuzzy: bool = False) -> List[Tuple[str, str]]:
    if isinstance(pattern, str):
        pattern = [pattern]
    if fuzzy:
        pattern = [f'*{p}*' for p in pattern]
    files_list = []
    for root, _, files in os.walk(path):
        for name in files:
            for p in pattern:
                if fnmatch.fnmatch(name.lower(), p.lower()):
                    files_list.append((name[:-3], os.path.join(root, name)))
                    break

    return sorted(files_list, key=lambda x: x[0])


def match_cfg_file(workdir: Union[str, List[str]],
                   pattern: Union[str, List[str]]) -> List[Tuple[str, str]]:
    """Match the config file in workdir recursively given the pattern.

    Additionally, if the pattern itself points to an existing file, it will be
    directly returned.
    """
    def _mf_with_multi_workdirs(workdir, pattern, fuzzy=False):
        if isinstance(workdir, str):
            workdir = [workdir]
        files = []
        for wd in workdir:
            files += match_files(wd, pattern, fuzzy=fuzzy)
        return files

    if isinstance(pattern, str):
        pattern = [pattern]
    pattern = [p + '.py' if not p.endswith('.py') else p for p in pattern]
    files = _mf_with_multi_workdirs(workdir, pattern, fuzzy=False)
    if len(files) != len(pattern):
        nomatched = []
        ambiguous = []
        ambiguous_return_list = []
        err_msg = ('The provided pattern matches 0 or more than one '
                   'config. Please verify your pattern and try again. '
                   'You may use tools/list_configs.py to list or '
                   'locate the configurations.\n')
        for p in pattern:
            files_ = _mf_with_multi_workdirs(workdir, p, fuzzy=False)
            if len(files_) == 0:
                nomatched.append([p[:-3]])
            elif len(files_) > 1:
                ambiguous.append([p[:-3], '\n'.join(f[1] for f in files_)])
                ambiguous_return_list.append(files_[0])
        if nomatched:
            table = [['Not matched patterns'], *nomatched]
            err_msg += tabulate.tabulate(table,
                                         headers='firstrow',
                                         tablefmt='psql')
        if ambiguous:
            table = [['Ambiguous patterns', 'Matched files'], *ambiguous]
            warning_msg = 'Found ambiguous patterns, using the first matched config.\n'
            warning_msg += tabulate.tabulate(table,
                                         headers='firstrow',
                                         tablefmt='psql')
            logger.warning(warning_msg)
            return ambiguous_return_list

        raise ValueError(err_msg)
    return files

def search_configs_from_args(args):
    """Get the config object given args.
    """
    logger.info('Searching configs...')
    table = [["Task Type", "Task Name", "Config File Path"]]
     # parse model args
    if args.models:
        models = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        models_dir = [
            os.path.join(args.config_dir, 'models'),
            os.path.join(default_configs_dir, './models'),
        ]
        for model_arg in args.models:
            for model in match_cfg_file(models_dir, [model_arg]):
                table.append(["--models", model[0], os.path.abspath(model[1])])

    # parse dataset args
    if args.datasets:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        datasets_dir = [
            os.path.join(args.config_dir, 'datasets'),
            os.path.join(args.config_dir, 'dataset_collections'),
            os.path.join(default_configs_dir, './datasets'),
            os.path.join(default_configs_dir, './dataset_collections')
        ]
        for dataset_arg in args.datasets:
            if '/' in dataset_arg:
                dataset_name, dataset_suffix = dataset_arg.split('/', 1)
            else:
                dataset_name = dataset_arg

            for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                table.append(["--datasets", dataset[0], os.path.abspath(dataset[1])])

    # parse summarizer args
    if args.summarizer:
        summarizer_arg = args.summarizer if args.summarizer is not None else 'example'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        summarizers_dir = [
            os.path.join(args.config_dir, 'summarizers'),
            os.path.join(default_configs_dir, './summarizers'),
        ]

        # Check if summarizer_arg contains '/'
        if '/' in summarizer_arg:
            # If it contains '/', split the string by '/'
            # and use the second part as the configuration key
            summarizer_file, summarizer_key = summarizer_arg.split('/', 1)
        else:
            # If it does not contain '/', keep the original logic unchanged
            summarizer_file = summarizer_arg

        s = match_cfg_file(summarizers_dir, [summarizer_file])[0]
        table.append(["--summarizer", s[0], os.path.abspath(s[1])])
    print(
        tabulate.tabulate(
            table,
            headers='firstrow',
            tablefmt="fancy_grid",  # 使用带边框的表格样式
            stralign="left",        # 文本列左对齐
            missingval="N/A",       # 处理空值
        )
    )