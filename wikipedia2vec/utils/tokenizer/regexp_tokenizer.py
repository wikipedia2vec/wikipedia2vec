import re
from typing import List, Tuple

from .base_tokenizer import BaseTokenizer


class RegexpTokenizer(BaseTokenizer):
    def __init__(self, rule=r"[\w\d]+"):
        self._rule = rule

        self._rule_obj = re.compile(rule, re.UNICODE)

    def _span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        return [obj.span() for obj in self._rule_obj.finditer(text)]

    def __reduce__(self):
        return (self.__class__, (self._rule,))
