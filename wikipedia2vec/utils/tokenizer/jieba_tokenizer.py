import logging
import re
from typing import List, Tuple

import jieba

from .base_tokenizer import BaseTokenizer

jieba.setLogLevel(logging.WARN)


class JiebaTokenizer(BaseTokenizer):
    def __init__(self):
        self._rule = re.compile(r"^\s*$")

    def _span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        return [(start, end) for (word, start, end) in jieba.tokenize(text) if not self._rule.match(word)]

    def __reduce__(self):
        return (self.__class__, tuple())
