# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

import logging
import re
from icu import Locale, BreakIterator

from .base_tokenizer cimport BaseTokenizer


cdef class ICUTokenizer(BaseTokenizer):
    cdef _locale
    cdef _rule
    cdef _breaker
    cdef _rule_obj

    def __init__(self, locale, rule=r'^[\w\d]+$'):
        self._locale = locale
        self._rule = rule

        self._breaker = BreakIterator.createWordInstance(Locale(locale))
        self._rule_obj = re.compile(rule, re.UNICODE)

    cdef list _span_tokenize(self, unicode text):
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
