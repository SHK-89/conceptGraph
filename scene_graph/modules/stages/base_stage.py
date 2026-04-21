from abc import ABC, abstractmethod


class BaseStage(ABC):

    @abstractmethod
    def run(self, *args, **kwargs):
        pass
