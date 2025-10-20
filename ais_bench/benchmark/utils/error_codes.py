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


# 初始化错误码管理器
error_manager = ErrorCodeManager()

# 应用层错误码
APPLICATION_LAYER_ERRORS = [
    # TaskManager相关
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task manager"), # TMAN-UNK-001

    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 1, "command miss required argument"), # TMAN-CMD-001
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.COMMAND, 2, "invalid argument value in command"), # TMAN-CMD-002

    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 1, "invaild syntax in config content"), # TMAN-CFG-001
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 2, "config content miss required param"), # TMAN-CFG-002
    BaseErrorCode(ErrorModule.TASK_MANAGER, ErrorType.CONFIG, 3, "type error in config param"), # TMAN-CFG-003

    # Partitioner相关
    BaseErrorCode(ErrorModule.PARTITIONER, ErrorType.UNKNOWN, 1, "unknown error of partitioner"), # PARTI-UNK-001

    # Summary相关
    BaseErrorCode(ErrorModule.SUMMARY, ErrorType.UNKNOWN, 1, "unknown error of summary"), # SUMM-UNK-001

]

# 业务逻辑层错误码
BUSINESS_LOGIC_LAYER_ERRORS = [
    # Runner 相关
    BaseErrorCode(ErrorModule.RUNNER, ErrorType.UNKNOWN, 1, "unknown error of runner"), # RUNN-UNK-001

    # TaskMonitor相关
    BaseErrorCode(ErrorModule.TASK_MONITOR, ErrorType.UNKNOWN, 1, "unknown error of task monitor"), # TMON-UNK-001

    # TaskStateManager相关
    BaseErrorCode(ErrorModule.TASK_STATUS_MANAGER, ErrorType.UNKNOWN, 1, "unknown error of task state manager"), # TSMAN-UNK-001

    # 推理Task相关
    BaseErrorCode(ErrorModule.TASK_INFER, ErrorType.UNKNOWN, 1, "unknown error of infer task"), # TINFER-UNK-001

    # 评估Task相关
    BaseErrorCode(ErrorModule.TASK_EVALUATE, ErrorType.UNKNOWN, 1, "unknown error of evaluate task"), # TEVAL-UNK-001

]

# ICL层错误码
ICL_LAYER_ERRORS = [
    # icl_inferencer相关
    BaseErrorCode(ErrorModule.ICL_INFERENCER, ErrorType.UNKNOWN, 1, "unknown error of icl inferencer"), # ICLI-UNK-001

    # icl_evaluator相关
    BaseErrorCode(ErrorModule.ICL_EVALUATOR, ErrorType.UNKNOWN, 1, "unknown error of icl evaluator"), # ICLE-UNK-001

    # icl_retriever相关
    BaseErrorCode(ErrorModule.ICL_RETRIEVER, ErrorType.UNKNOWN, 1, "unknown error of icl retriever"), # ICLR-UNK-001

    # model相关
    BaseErrorCode(ErrorModule.MODEL, ErrorType.UNKNOWN, 1, "unknown error of model"), # MODEL-UNK-001

]


# 未知报错
OTHER_ERRORS = [
    # unknown
    BaseErrorCode(ErrorModule.UNKNOWN, ErrorType.UNKNOWN, 1, "unknown error"), # UNK-UNK-001

    # utils相关
    BaseErrorCode(ErrorModule.UTILS, ErrorType.UNKNOWN, 1, "unknown error of utils"), # UTILS-UNK-001

    BaseErrorCode(ErrorModule.UTILS, ErrorType.MATCH, 1, "match config file failed"), # UTILS-MATCH-001
    BaseErrorCode(ErrorModule.UTILS, ErrorType.CONFIG, 1, "synthetic dataset miss required param"), # UTILS-CFG-001
]


# 注册所有错误码
for error in APPLICATION_LAYER_ERRORS + BUSINESS_LOGIC_LAYER_ERRORS + ICL_LAYER_ERRORS + OTHER_ERRORS:
    error_manager.register(error)
