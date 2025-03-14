from typing import Callable, List, Optional, Type, Union

from mmengine.registry import METRICS as MMENGINE_METRICS
from mmengine.registry import Registry as OriginalRegistry


class Registry(OriginalRegistry):

    # override the default force behavior
    def register_module(
            self,
            name: Optional[Union[str, List[str]]] = None,
            force: bool = True,
            module: Optional[Type] = None) -> Union[type, Callable]:
        return super().register_module(name, force, module)


PARTITIONERS = Registry('partitioner', locations=['ais_bench.benchmark.partitioners'])
RUNNERS = Registry('runner', locations=['ais_bench.benchmark.runners'])
TASKS = Registry('task', locations=['ais_bench.benchmark.tasks'])
MODELS = Registry('model', locations=['ais_bench.benchmark.models'])
# TODO: LOAD_DATASET -> DATASETS
LOAD_DATASET = Registry('load_dataset', locations=['ais_bench.benchmark.datasets'])
TEXT_POSTPROCESSORS = Registry(
    'text_postprocessors', locations=['ais_bench.benchmark.utils.text_postprocessors'])
DICT_POSTPROCESSORS = Registry(
    'dict_postprocessors', locations=['ais_bench.benchmark.utils.dict_postprocessors'])

EVALUATORS = Registry('evaluators', locations=['ais_bench.benchmark.evaluators'])

ICL_INFERENCERS = Registry('icl_inferencers',
                           locations=['ais_bench.benchmark.openicl.icl_inferencer'])
ICL_RETRIEVERS = Registry('icl_retrievers',
                          locations=['ais_bench.benchmark.openicl.icl_retriever'])
ICL_DATASET_READERS = Registry(
    'icl_dataset_readers',
    locations=['ais_bench.benchmark.openicl.icl_dataset_reader'])
ICL_PROMPT_TEMPLATES = Registry(
    'icl_prompt_templates',
    locations=['ais_bench.benchmark.openicl.icl_prompt_template'])
ICL_EVALUATORS = Registry('icl_evaluators',
                          locations=['ais_bench.benchmark.openicl.icl_evaluator'])
METRICS = Registry('metric',
                   parent=MMENGINE_METRICS,
                   locations=['ais_bench.benchmark.metrics'])
TOT_WRAPPER = Registry('tot_wrapper', locations=['ais_bench.benchmark.datasets'])


def build_from_cfg(cfg):
    """A helper function that builds object with MMEngine's new config."""
    return PARTITIONERS.build(cfg)
