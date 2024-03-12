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

from ais_bench.evaluate.measurement.measurement import (AccuracyMeasurement, EditDistanceMeasurement,
                                                        BleuMeasurement, RougeMeasurement,
                                                        AbnormalStringRateMeasurement, DistinctMeasurement,
                                                        RelativeDistinctMeasurement,
                                                        RelativeAbnormalStringRateMeasurement)
from ais_bench.evaluate.common.log import logger
from ais_bench.evaluate.common.global_var import *

measurement_switch = {
    ACCURACY: AccuracyMeasurement,
    EDIT_DISTANCE: EditDistanceMeasurement,
    BLEU: BleuMeasurement,
    ROUGE: RougeMeasurement,
    ABNORMAL_STRING_RATE: AbnormalStringRateMeasurement,
    DISTINCT: DistinctMeasurement,
    RELATIVE_DISTINCT: RelativeDistinctMeasurement,
    RELATIVE_ABNORMAL_STRING_RATE: RelativeAbnormalStringRateMeasurement,

}

class MeasurementFactory():
    @staticmethod
    def get(measurement, **kwargs):
        if measurement_switch.get(measurement.strip()) is not None:
            measurement_instance = measurement_switch.get(measurement.strip())(**kwargs)
        else:
            logger.error(f"Measurement {measurement} is not supported. "
                         f"Currently only {', '.join(list(measurement_switch.keys()))} are supported.")
            measurement_instance = None
        return measurement_instance

measurement_factory = MeasurementFactory()