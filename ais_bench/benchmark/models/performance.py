from typing import Dict, Optional

from ais_bench.benchmark.models.base import BaseModel


class PerformanceModel(BaseModel):
    def __init__(
        self,
        path: str,
        max_seq_len: int = 2048,
        tokenizer_only: bool = False,
        meta_template: Optional[Dict] = None,
        generation_kwargs: Optional[Dict] = dict(),
        sync_rank: bool = False,
    ) -> None:
        super().__init__(
            path,
            max_seq_len,
            tokenizer_only,
            meta_template,
            generation_kwargs,
            sync_rank,
        )
        self.do_performance = False

    def handle_perf_result(self, output_filepath, output_filename):
        pass

    def set_performance(self):
        self.do_performance = True