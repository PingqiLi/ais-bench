from typing import Optional
from ais_bench.benchmark.utils.error_codes import error_manager
from ais_bench.benchmark.utils.logger import get_formatted_log_content

class AISBenchBaseException(Exception):
    def __init__(self, error_str: str,
                 message: Optional[str] = None):
        """
        Args:
            error_str (str): full code of error code
            message (Optional[str], optional): error message. Defaults to None.

        """
        error_code = error_manager.get(error_str)
        self.error_code_str = error_str
        if not error_code:
            raise ValueError(f"error_code {error_str} is not exist!")
        super().__init__(get_formatted_log_content(error_str, message))


class CommandError(AISBenchBaseException):
    pass


class ConfigError(AISBenchBaseException):
    pass


class FileMatchError(AISBenchBaseException):
    pass


class PerfResultCalcException(AISBenchBaseException):
    pass