from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class CPGInput:
    cut_L: float = 0.0 # Hz comming from model.cut
    cut_R: float = 0.0

@dataclass
class CPGOutput:
    mus_E_L: float = 0.0
    mus_F_L: float = 0.0
    mus_E_R: float = 0.0
    mus_F_R: float = 0.0
    t_ms: float = 0.0


class CPGAdapter(ABC):
    @abstractmethod
    def update(self, cpg_in: CPGInput):
        pass

    @abstractmethod
    def step(self) -> CPGOutput:
        pass

    @abstractmethod
    def build_cpg(self):
        pass

    @abstractmethod
    def get_logs(self):
        pass