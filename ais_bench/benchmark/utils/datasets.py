import os
import json
import zipfile
import hashlib
import urllib.request

from .logging import get_logger


def get_cache_dir(default_dir):
    return os.environ.get('AIS_BENCH_DATASETS_CACHE', default_dir)


def get_data_path(dataset_path: str, local_mode: bool = True):
    """return dataset id when getting data from ModelScope/HuggingFace repo, otherwise just
    return local path as is.

    Args:
        dataset_path (str): data path
        local_mode (bool): whether to use local path or
            ModelScope/HuggignFace repo
    """
    # update the path with CACHE_DIR
    default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../") # site-package
    cache_dir = get_cache_dir(default_dir)

    # For absolute path customized by the users, will not auto download dataset
    if dataset_path.startswith('/'):
        return dataset_path

    # For relative path, with CACHE_DIR
    if local_mode:
        local_path = os.path.join(cache_dir, dataset_path)

        if not os.path.exists(local_path):
            readme_path = os.path.join(default_dir, "README.md")
            raise FileExistsError(f"Dataset path: {local_path} is not exist! " +
                                  "Please check section \"--datasets支持的数据集\" of " +
                                  f"{readme_path} to check how to prepare supported datasets.")
        else:
            return local_path
    else:
        raise TypeError('Customized dataset path type is not a absolute path!')