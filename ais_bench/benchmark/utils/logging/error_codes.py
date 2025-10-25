from enum import Enum, unique
from typing import Dict, Any, Optional

@unique
class ErrorModule(Enum):
    TASK_MANAGER = "TMAN"                        # TaskManager
    PARTITIONER = "PARTI"                        # Partitioner
    SUMMARY = "SUMM"                             # Summary
    RUNNER = "RUNNER"                            # Runner
    TASK_INFER = "TINFER"                        # inference Task
    TASK_EVALUATE = "TEVAL"                      # evaluate Task
    TASK_MONITOR = "TMON"                        # TaskMonitor
    TASK_STATUS_MANAGER = "TSMAN"                # TaskStateManager
    ICL_INFERENCER = "ICLI"                      # icl_inferencer
    ICL_EVALUATOR = "ICLE"                       # icl_evaluator
    ICL_RETRIEVER = "ICLR"                       # icl_retriever
    MODEL = "MODEL"                              # model
    UTILS = "UTILS"                              # other utils func
    UNKNOWN = "UNK"                              # unknown module


@unique
class ErrorType(Enum):
    UNKNOWN = "UNK"     # unknown error type
    COMMAND = "CMD"     # command error type
    CONFIG = "CFG"     # config error type
    MATCH = "MATCH"     # pattern match error type
    FILE = "FILE"     # file error type

class BaseErrorCode:
    FAQ_BASE_URL = "https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/faqs/error_codes.html#"

    def __init__(self, module: ErrorModule, err_type: ErrorType, code: int,
                 message: str):
        """
        Args:
            module (ErrorModule): error module
            err_type (ErrorType): error type
            code (int): error code number
            message (str): error message

        """
        self.module = module
        self.err_type = err_type
        self.code = code
        self.message = message
        self.faq_url = self.FAQ_BASE_URL + self.heading_id

    @property
    def full_code(self) -> str:
        return f"{self.module.value}-{self.err_type.value}-{self.code:03d}"
    @property
    def heading_id(self) -> str:
        return self.full_code.lower()

    def __str__(self) -> str:
        return f"{self.full_code}: {self.message}"



class ErrorCodeManager:
    def __init__(self):
        self._error_codes: Dict[str, BaseErrorCode] = {}

    def register(self, error_code: BaseErrorCode) -> None:
        if error_code.full_code in self._error_codes:
            raise ValueError(f"error code {error_code.full_code} is exist!")
        self._error_codes[error_code.full_code] = error_code

    def get(self, full_code: str) -> Optional[BaseErrorCode]:
        return self._error_codes.get(full_code)

    def list_all(self) -> Dict[str, BaseErrorCode]:
        return self._error_codes.copy()


# init error code manager
error_manager = ErrorCodeManager()

# regist application layer errors
APPLICATION_LAYER_ERRORS = [
    # TaskManager
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task manager"), # TMAN-UNK-001

    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 1, "command miss required argument"), # TMAN-CMD-001
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 2, "invalid argument value in command"), # TMAN-CMD-002

    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 1, "invaild syntax in config content"), # TMAN-CFG-001
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 2, "config content miss required param"), # TMAN-CFG-002
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 3, "type error in config param"), # TMAN-CFG-003

    # Partitioner
    BaseErrorCode(ErrorModule.PARTITIONER, ErrorType.UNKNOWN, 1, "unknown error of partitioner"), # PARTI-UNK-001
    BaseErrorCode(ErrorModule.PARTITIONER, ErrorType.FILE, 1, "out dir permission denied"), # PARTI-FILE-001

    # Summary
    BaseErrorCode(ErrorModule.SUMMARY, ErrorType.UNKNOWN, 1, "unknown error of summary"), # SUMM-UNK-001

]

# regist business logic layer errors
BUSINESS_LOGIC_LAYER_ERRORS = [
    # Runner
    BaseErrorCode(ErrorModule.RUNNER, ErrorType.UNKNOWN, 1, "unknown error of runner"), # RUNNER-UNK-001

    # TaskMonitor
    BaseErrorCode(ErrorModule.TASK_MONITOR, ErrorType.UNKNOWN, 1, "unknown error of task monitor"), # TMON-UNK-001

    # TaskStateManager
    BaseErrorCode(ErrorModule.TASK_STATUS_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task state manager"), # TSMAN-UNK-001

    # Infer Task
    BaseErrorCode(ErrorModule.TASK_INFER, ErrorType.UNKNOWN, 1, "unknown error of infer task"), # TINFER-UNK-001

    # Eval Task
    BaseErrorCode(ErrorModule.TASK_EVALUATE, ErrorType.UNKNOWN, 1, "unknown error of evaluate task"), # TEVAL-UNK-001

]

# regist icl layer errors
ICL_LAYER_ERRORS = [
    # icl_inferencer
    BaseErrorCode(ErrorModule.ICL_INFERENCER, ErrorType.UNKNOWN, 1, "unknown error of icl inferencer"), # ICLI-UNK-001

    # icl_evaluator
    BaseErrorCode(ErrorModule.ICL_EVALUATOR, ErrorType.UNKNOWN, 1, "unknown error of icl evaluator"), # ICLE-UNK-001

    # icl_retriever
    BaseErrorCode(ErrorModule.ICL_RETRIEVER, ErrorType.UNKNOWN, 1, "unknown error of icl retriever"), # ICLR-UNK-001

    # model
    BaseErrorCode(ErrorModule.MODEL, ErrorType.UNKNOWN, 1, "unknown error of model"), # MODEL-UNK-001

]


# regist other errors
OTHER_ERRORS = [
    # unknown
    BaseErrorCode(ErrorModule.UNKNOWN, ErrorType.UNKNOWN, 1, "unknown error"), # UNK-UNK-001

    # utils
    BaseErrorCode(ErrorModule.UTILS, ErrorType.UNKNOWN, 1, "unknown error of utils"), # UTILS-UNK-001

    BaseErrorCode(ErrorModule.UTILS, ErrorType.MATCH, 1, "match config file failed"), # UTILS-MATCH-001
    BaseErrorCode(ErrorModule.UTILS, ErrorType.CONFIG, 1, "synthetic dataset miss required param"), # UTILS-CFG-001
]


# regist all errors
for error in APPLICATION_LAYER_ERRORS + BUSINESS_LOGIC_LAYER_ERRORS + ICL_LAYER_ERRORS + OTHER_ERRORS:
    error_manager.register(error)


# error code consts
class TMAN_CODES:
    UNKNOWN_ERROR = "TMAN-UNK-001" # unknown error of task manager
    CMD_MISS_REQUIRED_ARG = "TMAN-CMD-001" # command miss required argument
    INVALID_ARG_VALUE_IN_CMD = "TMAN-CMD-002" # invalid argument value in command
    INVAILD_SYNTAX_IN_CFG_CONTENT = "TMAN-CFG-001" # invaild syntax in config content
    CFG_CONTENT_MISS_REQUIRED_PARAM = "TMAN-CFG-002" # config content miss required param
    TYPE_ERROR_IN_CFG_PARAM = "TMAN-CFG-003" # type error in config param


class PARTI_CODES:
    UNKNOWN_ERROR = "PARTI-UNK-001" # unknown error of partitioner
    OUT_DIR_PERMISSION_DENIED = "PARTI-FILE-001" # out dir permission denied


class SUMM_CODES:
    UNKNOWN_ERROR = "SUMM-UNK-001" # unknown error of summary


class RUNNER_CODES:
    UNKNOWN_ERROR = "RUNNER-UNK-001" # unknown error of runner


class TMON_CODES:
    UNKNOWN_ERROR = "TMON-UNK-001" # unknown error of task monitor


class TSMAN_CODES:
    UNKNOWN_ERROR = "TSMAN-UNK-001" # unknown error of task state manager


class TINFER_CODES:
    UNKNOWN_ERROR = "TINFER-UNK-001" # unknown error of infer task


class TEVAL_CODES:
    UNKNOWN_ERROR = "TEVAL-UNK-001" # unknown error of evaluate task


class ICLI_CODES:
    UNKNOWN_ERROR = "ICLI-UNK-001" # unknown error of icl inferencer


class ICLE_CODES:
    UNKNOWN_ERROR = "ICLE-UNK-001" # unknown error of icl evaluator


class ICLR_CODES:
    UNKNOWN_ERROR = "ICLR-UNK-001" # unknown error of icl retriever


class MODEL_CODES:
    UNKNOWN_ERROR = "MODEL-UNK-001" # unknown error of model


class UNK_CODES:
    UNKNOWN_ERROR = "UNK-UNK-001" # unknown error of utils


class UTILS_CODES:
    UNKNOWN_ERROR = "UTILS-UNK-001" # unknown error of utils
    MATCH_1 = "UTILS-MATCH-001" # match config file failed
    SYNTHETIC_DS_MISS_REQUIRED_PARAM = "UTILS-CFG-001" # synthetic dataset miss required param