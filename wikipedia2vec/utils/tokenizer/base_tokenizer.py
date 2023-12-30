from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from .token import Token


class BaseTokenizer(metaclass=ABCMeta):
    def tokenize(self, text: str) -> List[Token]:
        return [Token(text[start:end], start, end) for (start, end) in self._span_tokenize(text)]

    @abstractmethod
    def _span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        raise NotImplementedError()
