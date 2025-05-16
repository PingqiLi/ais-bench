from abc import abstractmethod, ABC

class BasePerfMetricCalculator(ABC):
    def __init__(self, result: dict):
        self.result = result

    @abstractmethod
    def get_common_res(self):
        return {}

    @abstractmethod
    def save_performance(self, out_path: str):
        pass

    @abstractmethod
    def calculate(self):
        pass
