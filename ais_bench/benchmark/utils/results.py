import json
import orjson
import h5py
from tqdm import tqdm
import numpy as np
import fcntl
import os

MAX_H5_CHUNK_SIZE = 5000


def safe_write(results_dict: dict, filename):
    """
    use fcntl file lock to implement mutual exclusion writing
    """
    with open(filename, "a", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            for _, result in results_dict.items():
                f.write(json.dumps(result) + "\n")
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def dump_results_dict(results_dict, filename, formatted=True):
    with open(filename, "w", encoding="utf-8") as json_file:
        if formatted:
            json.dump(results_dict, json_file, indent=4, ensure_ascii=False)
        else:
            json.dump(results_dict, json_file, ensure_ascii=False)


def fast_dump_results_dict(results_dict, filename):
    with open(filename, "wb") as f:
        f.write(orjson.dumps(results_dict))


def dump_list_as_h5(data_list, h5_file, data_type=np.float32):
    total_rows = len(data_list)
    chunk_size = MAX_H5_CHUNK_SIZE if total_rows >= MAX_H5_CHUNK_SIZE else total_rows
    with h5py.File(h5_file, "w") as f:
        dt = h5py.vlen_dtype(data_type)
        dset = f.create_dataset(
            "arrays", (total_rows,), dtype=dt, compression="gzip", chunks=(chunk_size,)
        )

        # write in chunks (with progress bar)
        for i in tqdm(range(0, total_rows, chunk_size), desc="Dumping data to h5"):
            end_idx = min(i + chunk_size, total_rows)
            chunk = data_list[i:end_idx]
            for j, arr in enumerate(chunk):
                dset[i + j] = arr


def load_from_h5(h5_file, group_name="arrays"):
    with h5py.File(h5_file, "r") as f:
        group = f[group_name]
        data = {uid: ds[()] for uid, ds in group.items()}
    return data