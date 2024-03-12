#!/usr/bin/env bash
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

CUR_PATH=$(cd "$(dirname "$0")";pwd)

build_evaluate() {
    [ -d ${CUR_PATH}/dist ] && rm -rf ${CUR_PATH}/dist
    cd ${CUR_PATH}
    python3 setup.py bdist_wheel

    [ -d ${CUR_PATH}/output ] && rm -rf ${CUR_PATH}/output
    mkdir ${CUR_PATH}/output
    cp ${CUR_PATH}/dist/*.whl ${CUR_PATH}/output
    cp ${CUR_PATH}/README.md ${CUR_PATH}/output
}


build_evaluate
if [ "$1" == "install" ];then
    pip3 install ${CUR_PATH}/dist/*.whl --force-reinstall
fi
