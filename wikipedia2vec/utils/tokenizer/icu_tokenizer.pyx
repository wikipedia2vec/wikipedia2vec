# -*- coding: utf-8 -*-
# cython: profile=False

import logging
import re
from icu import Locale, BreakIterator

from .base_tokenizer cimport BaseTokenizer


cdef class ICUTokenizer(BaseTokenizer):
    cdef _breaker
    cdef _rule

    def __init__(self, locale, rule=r'^[\w\d]+$'):
        self._breaker = BreakIterator.createWordInstance(Locale(locale))
        self._rule = re.compile(rule, re.UNICODE)

    cdef list _span_tokenize(self, unicode text):
        self._breaker.setText(text)

        ret = []
        start = self._breaker.first()
        for end in self._breaker:
            if self._rule.match(text[start:end]):
                ret.append((start, end))
            start = end

        return ret
