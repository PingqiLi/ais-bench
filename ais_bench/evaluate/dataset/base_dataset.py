# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
import hashlib
import json
from abc import abstractmethod, ABCMeta

from ais_bench.evaluate.common.log import logger
from ais_bench.evaluate.common.path_check import ms_open, MAX_SIZE_LIMITED_NORMAL_FILE


class BaseDataset(metaclass=ABCMeta):
    def __init__(self, dataset_name, dataset_path=None, shot=0) -> None:
        self.dataset_name = dataset_name
        self.shot = shot
        self.dataset_path = dataset_path
        self.load()

    @staticmethod
    def hash(file_path):
        hasher = hashlib.sha256()
        with ms_open(file_path, mode="rb", max_size=MAX_SIZE_LIMITED_NORMAL_FILE) as file:
            for chunk in iter(lambda: file.read(4096), b''):
                hasher.update(chunk)
        hash_value = hasher.hexdigest()
        return hash_value

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __iter__(self):
        raise NotImplementedError

    @abstractmethod
    def __next__(self):
        raise NotImplementedError

    @abstractmethod
    def load(self):
        raise NotImplementedError

    def _download(self):
        parent_path = os.path.dirname(os.path.realpath(__file__))
        download_sh_path = os.path.join(parent_path, "download.sh")
        os.chmod(download_sh_path, 0o550)
        result = subprocess.run([download_sh_path, self.dataset_name], check=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode == 0:
            self.dataset_path = os.path.join(parent_path, self.dataset_name)
        else:
            logger.error("please download the dataset.")
            raise ValueError

    def _check(self):
        parent_path = os.path.dirname(os.path.realpath(__file__))
        hash_json_path = os.path.join(parent_path, f"{self.dataset_name}_sha256.json")
        with ms_open(hash_json_path, max_size=MAX_SIZE_LIMITED_NORMAL_FILE) as file:
            hash_dict = json.load(file)
        dataset_dir = self.dataset_path
        for relative_path, true_hash in hash_dict.items():
            file_path = os.path.join(dataset_dir, relative_path)
            if (self.hash(file_path) != true_hash):
                logger.error("dataset verification failed: file hash value different")
                raise ValueError