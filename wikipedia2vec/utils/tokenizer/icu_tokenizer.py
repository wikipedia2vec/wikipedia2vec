import re
from typing import List, Tuple

from icu import Locale, BreakIterator

from .base_tokenizer import BaseTokenizer


class ICUTokenizer(BaseTokenizer):
    def __init__(self, locale: str, rule: str = r"^[\w\d]+$"):
        self._locale = locale
        self._rule = rule

        self._breaker = BreakIterator.createWordInstance(Locale(locale))
        self._rule_obj = re.compile(rule, re.UNICODE)

    def _span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        self._breaker.setText(text)

        ret = []
        start = self._breaker.first()
        for end in self._breaker:
            if self._rule_obj.match(text[start:end]):
                ret.append((start, end))
            start = end

        return ret

    def __reduce__(self):
        return (self.__class__, (self._locale, self._rule))
