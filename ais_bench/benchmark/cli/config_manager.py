
import os
import os.path as osp
import tabulate
from mmengine.config import Config, DictAction

from ais_bench.benchmark.utils import get_logger
from ais_bench.benchmark.datasets.custom import make_custom_dataset_config
from ais_bench.benchmark.utils.run import (function_call_task_check, try_fill_in_custom_cfgs, match_cfg_file, )


class ConfigManager:
    def __init__(self, args):
        self.args = args
        self.logger = get_logger(log_level='DEBUG' if args.debug else 'INFO')

    def search_configs_location(self):
        """Get the config object given args.
        """
        self.logger.info('Searching configs...')
        self.table = [["Task Type", "Task Name", "Config File Path"]]
        if self.args.models:
           self._search_models_config()

        if self.args.datasets:
            self._search_datasets_config()

        if self.args.summarizer:
            self._search_summarizers_config()

        print( # origin print
            tabulate.tabulate(
                self.table,
                headers='firstrow',
                tablefmt="fancy_grid",
                stralign="left",
                missingval="N/A",
            )
        )

    def load_config(self, workflow):
        self.cfg = self._get_config_from_arg()
        self._update_and_init_work_dir()
        self._update_cfg_of_workflow(workflow)
        self._dump_and_reload_config()
        return self.cfg

    def _search_models_config(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        models_dir = [
            os.path.join(self.args.config_dir, 'models'),
            os.path.join(default_configs_dir, './models'),
        ]
        for model_arg in self.args.models:
            for model in match_cfg_file(models_dir, [model_arg]):
                self.table.append(["--models", model[0], os.path.abspath(model[1])])

    def _search_datasets_config(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        datasets_dir = [
            os.path.join(self.args.config_dir, 'datasets'),
            os.path.join(self.args.config_dir, 'dataset_collections'),
            os.path.join(default_configs_dir, './datasets'),
            os.path.join(default_configs_dir, './dataset_collections')
        ]
        for dataset_arg in self.args.datasets:
            if '/' in dataset_arg:
                dataset_name, dataset_suffix = dataset_arg.split('/', 1)
            else:
                dataset_name = dataset_arg

            for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                self.table.append(["--datasets", dataset[0], os.path.abspath(dataset[1])])

    def _search_summarizers_config(self):
        summarizer_arg = self.args.summarizer if self.args.summarizer is not None else 'example'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        summarizers_dir = [
            os.path.join(self.args.config_dir, 'summarizers'),
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
        self.table.append(["--summarizer", s[0], os.path.abspath(s[1])])

    def _get_config_from_arg(self):
        if self.args.config:
            config = Config.fromfile(self.args.config, format_python_code=False)
            config = try_fill_in_custom_cfgs(config)
            return config

        models = self._load_models_config()
        datasets = self._load_datasets_config()
        summarizer = self._load_summarizers_config()

        return Config(dict(models=models, datasets=datasets, summarizer=summarizer, cli_args=vars(self.args)), format_python_code=False)

    def _load_datasets_config(self):
        datasets = []
        if self.args.datasets:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            default_configs_dir = os.path.join(parent_dir, 'configs')
            datasets_dir = [
                os.path.join(self.args.config_dir, 'datasets'),
                os.path.join(self.args.config_dir, 'dataset_collections'),
                os.path.join(default_configs_dir, './datasets'),
                os.path.join(default_configs_dir, './dataset_collections')
            ]
            for dataset_arg in self.args.datasets:
                if '/' in dataset_arg:
                    dataset_name, dataset_suffix = dataset_arg.split('/', 1)
                    dataset_key_suffix = dataset_suffix
                else:
                    dataset_name = dataset_arg
                    dataset_key_suffix = '_datasets'

                for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                    self.logger.info(f'Loading {dataset[0]}: {dataset[1]}')
                    cfg = Config.fromfile(dataset[1])
                    for k in cfg.keys():
                        if k.endswith(dataset_key_suffix):
                            datasets += cfg[k]
        else:
            # custom dataset need to reconstruct
            dataset = {'path': self.args.custom_dataset_path}
            if self.args.custom_dataset_infer_method is not None:
                dataset['infer_method'] = self.args.custom_dataset_infer_method
            if self.args.custom_dataset_data_type is not None:
                dataset['data_type'] = self.args.custom_dataset_data_type
            if self.args.custom_dataset_meta_path is not None:
                dataset['meta_path'] = self.args.custom_dataset_meta_path
            dataset = make_custom_dataset_config(dataset)
            datasets.append(dataset)
        return datasets

    def _load_models_config(self):
        if not self.args.models:
            raise ValueError('You must specify a config file path, or specify --models and --datasets.')
        models = []
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        models_dir = [
            os.path.join(self.args.config_dir, 'models'),
            os.path.join(default_configs_dir, './models'),

        ]
        if self.args.models:
            for model_arg in self.args.models:
                for model in match_cfg_file(models_dir, [model_arg]):
                    self.logger.info(f'Loading {model[0]}: {model[1]}')
                    cfg = Config.fromfile(model[1])
                    if 'models' not in cfg:
                        raise ValueError(f'Config file {model[1]} does not contain "models" field')
                    models += cfg['models']
        else:
            raise ValueError('You must specify "--models"')
        return models

    def _load_summarizers_config(self):
        # parse summarizer args
        summarizer_arg = self.args.summarizer if self.args.summarizer is not None else 'example'
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        summarizers_dir = [
            os.path.join(self.args.config_dir, 'summarizers'),
            os.path.join(default_configs_dir, './summarizers'),

        ]

        # Check if summarizer_arg contains '/'
        if '/' in summarizer_arg:
            # If it contains '/', split the string by '/'
            # and use the second part as the configuration key
            summarizer_file, summarizer_key = summarizer_arg.split('/', 1)
        else:
            # If it does not contain '/', keep the original logic unchanged
            summarizer_key = 'summarizer'
            summarizer_file = summarizer_arg

        s = match_cfg_file(summarizers_dir, [summarizer_file])[0]
        self.logger.info(f'Loading {s[0]}: {s[1]}')
        cfg = Config.fromfile(s[1])
        # Use summarizer_key to retrieve the summarizer definition
        # from the configuration file
        summarizer = cfg[summarizer_key]
        return summarizer

    def _update_and_init_work_dir(self):
        if self.args.work_dir is not None:
            self.cfg['work_dir'] = self.args.work_dir
        else:
            self.cfg.setdefault('work_dir', os.path.join('outputs', 'default'))

        # cfg_time_str defaults to the current time
        self.cfg_time_str = dir_time_str = self.args.dir_time_str

        if self.args.reuse:
            if self.args.reuse == 'latest':
                if not os.path.exists(self.cfg.work_dir) or not os.listdir(
                        self.cfg.work_dir):
                    self.logger.warning('No previous experiment results found to reuse.')
                else:
                    dirs = os.listdir(self.cfg.work_dir)
                    dir_time_str = sorted(dirs)[-1]
            else:
                dir_time_str = self.args.reuse
            self.args.dir_time_str = dir_time_str
            self.logger.info(f'Reusing experiements from {dir_time_str}')

        # update "actual" work_dir
        self.cfg['work_dir'] = osp.join(self.cfg.work_dir, dir_time_str)
        current_workdir = self.cfg['work_dir']
        self.logger.info(f'Current exp folder: {current_workdir}')

        os.makedirs(osp.join(self.cfg.work_dir, 'configs'), exist_ok=True)

    def _update_cfg_of_workflow(self, workflow):
        for work in workflow:
            self.cfg = work.update_cfg(self.cfg)

    def _dump_and_reload_config(self):
        # dump config
        output_config_path = osp.join(self.cfg.work_dir, 'configs',
                                    f'{self.cfg_time_str}_{os.getpid()}.py')
        self.cfg.dump(output_config_path)
        # eval nums set
        if (self.args.num_prompts and self.args.num_prompts < 0) or self.args.num_prompts == 0:
            raise ValueError("Num Prompts must be a positive integer greater than 0.")
        self.cfg['num_prompts'] = self.args.num_prompts
        # Config is intentally reloaded here to avoid initialized
        # types cannot be serialized
        self.cfg = Config.fromfile(output_config_path, format_python_code=False)

        # check if the tasks all function call tasks
        function_call_task_check(self.cfg, self.args.merge_ds)