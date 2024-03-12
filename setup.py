# Copyright (c) Huawei Technologies Co., Ltd. 2024. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from setuptools import setup, find_packages
import sys
import platform

with open('requirements.txt', encoding='utf-8') as f:
    required = f.read().splitlines()

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name="ais_bench_evaluate",
    version="2.0",
    description="ais_bench evaluate",
    long_description=long_description,
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    package_data={ "ais_bench.evaluate.dataset":["*.json", "*.sh"], }
)