import re
from abc import abstractmethod, ABC

DEFAULT_STATS = [
    "Average", "Min", "Max", "Median", "P75", "P90", "P99",
]
MAX_STATS_LEN = 8
PERCENTAGE_PATTERN = r'^P(0*[1-9]\d{0,1})$' # P1 ~ P99

def is_legal_percentage_str(stat):
    return re.match(PERCENTAGE_PATTERN, stat)

class BasePerfMetricCalculator(ABC):
    def __init__(self, perf_details: dict):
        self.perf_details = perf_details

    @abstractmethod
    def get_common_res(self):
        return {}

    @abstractmethod
    def save_performance(self, out_path: str):
        pass

    @abstractmethod
    def calculate(self):
        pass
    
    def extract_success_item(self, requests: dict):
        is_success = requests.get("is_success", [])
        for key, value in requests.items():
            if key == "is_success":
                continue
            if isinstance(value, list):
                value = [v for i, v in enumerate(value) if is_success[i]]
            else:
                value = value if is_success else []
            requests[key] = value
        return requests