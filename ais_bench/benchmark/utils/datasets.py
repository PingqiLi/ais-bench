import os
import csv
import json
import zipfile
import hashlib
import urllib.request
import random

from typing import List, Any, Optional, Callable, Dict, Union, Type, Tuple

from .logging import get_logger
logger = get_logger()

def get_cache_dir(default_dir):
    # TODO Add any necessary supplementary information for here
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


def get_sample_data(data_list: list, sample_mode: str = "default", request_count: int = 0):
    if not request_count:
        logger.info("If u do not provide 'request_count' when using custom-dataset sampling feature, "
                       "we will sample all available data by default.")
        sample_index = len(data_list)
    elif request_count > len(data_list):
        repeat_times = (request_count // len(data_list)) + (1 if request_count % len(data_list) != 0 else 0)
        data_list = (data_list * repeat_times)[:request_count]
        sample_index = request_count
    elif request_count < 0:
        raise ValueError("The 'request_count' is negative, we only support positive integer.")
    else:
        sample_index = request_count
    # sampling data
    if sample_mode == "default":
        return data_list[:sample_index]
    elif sample_mode == "random":
        return random.sample(data_list, sample_index)
    elif sample_mode == "shuffle":
        shuffle_data = data_list[:sample_index]
        random.shuffle(shuffle_data)
        return shuffle_data
    else:
        raise ValueError(f"Sample mode: {sample_mode} is not supported!")


def safe_load_json_file(
    path: str, 
    encoding: str = 'utf-8-sig', 
    expected_types: Union[Type, Tuple[Type, ...]] = (dict,)
) -> List[Any]:
    data: List[Any] = []
    try:
        with open(path, 'r', encoding=encoding) as f:
            for line_number, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    parsed = json.loads(line)
                    if not isinstance(parsed, expected_types):
                        type_names = "or".join(t.__name__ for t in expected_types)
                        raise TypeError(
                            f"Expected {type_names}, got {type(parsed).__name__}"
                        )
                    data.append(parsed)
                except (json.JSONDecodeError, TypeError) as e:
                    err_msg = (
                        f"JSON parse error at line {line_number}: {e}\n"
                        f"Problematic content: {line!r}"
                    )
                    raise ValueError(err_msg) from e
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading {path}: {e}")
    return data


def safe_load_csv_file(
    path: str, 
    encoding: str = 'utf-8-sig', 
    expected_types: Union[Type, Tuple[Type, ...]] = (str,)
) -> List[Dict[str, str]]:
    data: List[Dict[str, str]] = []
    try:
        with open(path, 'r', encoding=encoding) as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
                header = [col.strip() for col in header]
            except StopIteration:
                return data
            for row_number, row in enumerate(reader, 2):
                if not any(field.strip() for field in row):
                    continue # skip empty rows (all fields are empty or whitespace)
                try:
                    if len(row) != len(header):
                        raise ValueError(
                            f"Column count mismatch: expected {len(header)} columns, "
                            f"found {len(row)} columns"
                        )
                    row_dict = dict(zip(header, row))
                    for key, value in row_dict.items():
                        if not isinstance(value, expected_types):
                            type_names = "or".join(t.__name__ for t in expected_types)
                            actual_type = type(value).__name__
                            raise TypeError(
                                f"Value '{value}' for key '{key}' "
                                f"expected type {type_names}, got {actual_type}"
                            )
                    data.append(row_dict)
                except (ValueError, TypeError) as e:
                    err_msg = (
                        f"CSV processing error at line {row_number}: {e}\n"
                        f"Problematic row: {row!r}"
                    )
                    raise ValueError(err_msg) from e
    
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {path}")
    except csv.Error as e:
        raise ValueError(f"CSV format error: {e}")
    except Exception as e:
        raise ValueError(f"Unexpected error reading {path}: {e}")
    
    return data
