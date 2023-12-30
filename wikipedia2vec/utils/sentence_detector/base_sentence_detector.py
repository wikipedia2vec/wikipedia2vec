from abc import ABCMeta, abstractmethod
from typing import List

from .sentence import Sentence


class BaseSentenceDetector(metaclass=ABCMeta):
    @abstractmethod
    def detect_sentences(self, text: str) -> List[Sentence]:
        raise NotImplementedError()
