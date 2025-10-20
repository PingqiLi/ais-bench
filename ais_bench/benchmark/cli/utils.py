import sys
from datetime import datetime


def get_config_type(obj) -> str:
    if isinstance(obj, str):
        return obj
    return f"{obj.__module__}.{obj.__name__}"


def is_running_in_background():
    # check whether stdin and stdout are connected to TTY
    stdin_is_tty = sys.stdin.isatty()
    stdout_is_tty = sys.stdout.isatty()

    # if stdin and stdout are not connected to TTY, the script is running in background
    return not (stdin_is_tty and stdout_is_tty)


def get_current_time_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def fill_model_path_if_synthetic(model_cfg, dataset_cfg):
    data_type = get_config_type(dataset_cfg.get("type"))
    if (
        data_type == "ais_bench.benchmark.datasets.synthetic.SyntheticDataset"
        and dataset_cfg.get("config", {}).get("Type") == "tokenid"
    ):
        model_path = model_cfg.get("path")
        if not model_path:
            raise ValueError(
                "[path] in model config is required for synthetic dataset with tokenid type"
            )
        dataset_cfg.update({"model_path": model_path})
