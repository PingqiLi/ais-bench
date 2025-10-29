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
    CALCULATOR = "CALC"                        # calculator
    UTILS = "UTILS"                              # other utils func
    UNKNOWN = "UNK"                              # unknown module


@unique
class ErrorType(Enum):
    UNKNOWN = "UNK"     # unknown error type
    THIRD_PARTY = "THIRD_PARTY"     # third party error type
    IMPLEMENTATION = "IMPL"     # implementation error type
    COMMAND = "CMD"     # command error type
    CONFIG = "CFG"     # config error type
    MATCH = "MATCH"     # pattern match error type
    FILE = "FILE"     # file error type
    DATA = "DATA"     # data error type
    METRIC = "MTRC"     # metric error type
    TYPE = "TYPE"     # type error type
    PARAMETER = "PARAM"     # parameter error type
    
    
class BaseErrorCode:
    FAQ_BASE_URL = "https://ais-bench-benchmark.readthedocs.io/zh-cn/latest/faqs/error_codes.html#"

    def __init__(self, code_name: str, module: ErrorModule, err_type: ErrorType, code: int,
                 message: str):
        """
        Args:
            code_name (str): error code name (just for developer to check full_code)
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
        if code_name != self.full_code:
            raise ValueError(f"code_name {code_name} is not equal to full_code {self.full_code}")

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

# error code consts
class TMAN_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TMAN-UNK-001", ErrorModule.TASK_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task manager")
    CMD_MISS_REQUIRED_ARG = BaseErrorCode("TMAN-CMD-001", ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 1, "command miss required argument")
    INVALID_ARG_VALUE_IN_CMD = BaseErrorCode("TMAN-CMD-002", ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 2, "invalid argument value in command")
    INVAILD_SYNTAX_IN_CFG_CONTENT = BaseErrorCode("TMAN-CFG-001", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 1, "invaild syntax in config content")
    CFG_CONTENT_MISS_REQUIRED_PARAM = BaseErrorCode("TMAN-CFG-002", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 2, "config content miss required param")
    TYPE_ERROR_IN_CFG_PARAM = BaseErrorCode("TMAN-CFG-003", ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 3, "type error in config param")


class PARTI_CODES:
    UNKNOWN_ERROR = BaseErrorCode("PARTI-UNK-001", ErrorModule.PARTITIONER, ErrorType.UNKNOWN, 1, "unknown error of partitioner")
    OUT_DIR_PERMISSION_DENIED = BaseErrorCode("PARTI-FILE-001", ErrorModule.PARTITIONER, ErrorType.FILE, 1, "out dir permission denied")


class SUMM_CODES:
    UNKNOWN_ERROR = BaseErrorCode("SUMM-UNK-001", ErrorModule.SUMMARY, ErrorType.UNKNOWN, 1, "unknown error of summary")
    NOT_SUPPORTED_DATASET_TYPES = BaseErrorCode("SUMM-TYPE-001", ErrorModule.SUMMARY, ErrorType.TYPE, 1, "not support mixed dataset_abbr type")
    NO_PERF_DATA_FILE = BaseErrorCode("SUMM-FILE-001", ErrorModule.SUMMARY, ErrorType.FILE, 1, "can't find detail perf data file")
    DIFF_STRUCTURE_OF_PERF_DATA = BaseErrorCode("SUMM-MTRC-001", ErrorModule.SUMMARY, ErrorType.METRIC, 1, "different structure of perf data")


class RUNNER_CODES:
    UNKNOWN_ERROR = BaseErrorCode("RUNNER-UNK-001", ErrorModule.RUNNER, ErrorType.UNKNOWN, 1, "unknown error of runner")


class TMON_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TMON-UNK-001", ErrorModule.TASK_MONITOR, ErrorType.UNKNOWN, 1, "unknown error of task monitor")


class TSMAN_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TSMAN-UNK-001", ErrorModule.TASK_STATUS_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task state manager")


class TINFER_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TINFER-UNK-001", ErrorModule.TASK_INFER, ErrorType.UNKNOWN, 1, "unknown error of infer task")

class TEVAL_CODES:
    UNKNOWN_ERROR = BaseErrorCode("TEVAL-UNK-001", ErrorModule.TASK_EVALUATE, ErrorType.UNKNOWN, 1, "unknown error of evaluate task")


class ICLI_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLI-UNK-001", ErrorModule.ICL_INFERENCER, ErrorType.UNKNOWN, 1, "unknown error of icl inferencer")
    INVALID_PARAM_VALUE = BaseErrorCode("ICLI-PARAM-001", ErrorModule.ICL_INFERENCER, ErrorType.PARAMETER, 1, "invalid parameter value")
    IMPLEMENTATION_ERROR = BaseErrorCode("ICLI-IMPL-001", ErrorModule.ICL_INFERENCER, ErrorType.IMPLEMENTATION, 1, "not implemented error")
    FILE_OPERATION_ERROR = BaseErrorCode("ICLI-FILE-001", ErrorModule.ICL_INFERENCER, ErrorType.FILE, 1, "failed to write results files")
    
class ICLE_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLE-UNK-001", ErrorModule.ICL_EVALUATOR, ErrorType.UNKNOWN, 1, "unknown error of icl evaluator")
    PREDICTION_INVALID = BaseErrorCode("ICLE-DATA-001", ErrorModule.ICL_EVALUATOR, ErrorType.DATA, 1, "prediction invalid")
    REPLICATION_LENGTH_MISMATCH = BaseErrorCode("ICLE-DATA-002", ErrorModule.ICL_EVALUATOR, ErrorType.DATA, 2, "replication length mismatch")
    IMPLEMENTATION_ERROR = BaseErrorCode("ICLE-IMPL-001", ErrorModule.ICL_EVALUATOR, ErrorType.IMPLEMENTATION, 1, "not implemented error")

class ICLR_CODES:
    UNKNOWN_ERROR = BaseErrorCode("ICLR-UNK-001", ErrorModule.ICL_RETRIEVER, ErrorType.UNKNOWN, 1, "unknown error of icl retriever")


class MODEL_CODES:
    UNKNOWN_ERROR = BaseErrorCode("MODEL-UNK-001", ErrorModule.MODEL, ErrorType.UNKNOWN, 1, "unknown error of model")


class UNK_CODES:
    UNKNOWN_ERROR = BaseErrorCode("UNK-UNK-001", ErrorModule.UNKNOWN, ErrorType.UNKNOWN, 1, "unknown error")


class UTILS_CODES:
    UNKNOWN_ERROR = BaseErrorCode("UTILS-UNK-001", ErrorModule.UTILS, ErrorType.UNKNOWN, 1, "unknown error of utils")
    MATCH_CONFIG_FILE_FAILED = BaseErrorCode("UTILS-MATCH-001", ErrorModule.UTILS, ErrorType.MATCH, 1, "match config file failed")
    SYNTHETIC_DS_MISS_REQUIRED_PARAM = BaseErrorCode("UTILS-CFG-001", ErrorModule.UTILS, ErrorType.CONFIG, 1, "synthetic dataset miss required param")
    THIRD_PARTY_ERROR = BaseErrorCode("UTILS-THIRD_PARTY-001", ErrorModule.UTILS, ErrorType.THIRD_PARTY, 1, "third party error")
class CALC_CODES:
    UNKNOWN_ERROR = BaseErrorCode("CALC-UNK-001", ErrorModule.CALCULATOR, ErrorType.UNKNOWN, 1, "unknown error of calculator")
    INVALID_METRIC_DATA = BaseErrorCode("CALC-MTRC-001", ErrorModule.CALCULATOR, ErrorType.METRIC, 1, "invalid content of metric data")
    DUMPING_RESULT_FAILED = BaseErrorCode("CALC-FILE-001", ErrorModule.CALCULATOR, ErrorType.FILE, 1, "fail to dump result to file")
    ALL_REQUEST_DATAS_INVALID = BaseErrorCode("CALC-DATA-001", ErrorModule.CALCULATOR, ErrorType.DATA, 1, "all request datas are invalid")
    CAN_NOT_FIND_STABLE_STAGE = BaseErrorCode("CALC-DATA-002", ErrorModule.CALCULATOR, ErrorType.DATA, 2, "invalid response datas")


ERROR_CODES_CLASSES = [
    TMAN_CODES,
    PARTI_CODES,
    SUMM_CODES,
    RUNNER_CODES,
    TMON_CODES,
    TSMAN_CODES,
    TINFER_CODES,
    TEVAL_CODES,
    ICLI_CODES,
    ICLE_CODES,
    ICLR_CODES,
    MODEL_CODES,
    UNK_CODES,
    UTILS_CODES,
]

# init error code manager
error_manager = ErrorCodeManager()

# regist all errors
for error_codes_class in ERROR_CODES_CLASSES:
    for error_code in error_codes_class.__dict__.values():
        if isinstance(error_code, BaseErrorCode):
            error_manager.register(error_code)
