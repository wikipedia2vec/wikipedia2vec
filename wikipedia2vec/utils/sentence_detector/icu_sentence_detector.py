from typing import List

from icu import Locale, BreakIterator

from .base_sentence_detector import BaseSentenceDetector
from .sentence import Sentence


class ICUSentenceDetector(BaseSentenceDetector):
    def __init__(self, locale: str):
        self._locale = locale
        self._breaker = BreakIterator.createSentenceInstance(Locale(locale))

    def detect_sentences(self, text: str) -> List[Sentence]:
        self._breaker.setText(text)

        ret = []
        start = self._breaker.first()
        for end in self._breaker:
            ret.append(Sentence(text[start:end], start, end))
            start = end

        return ret

    def __reduce__(self):
        return (self.__class__, (self._locale,))
