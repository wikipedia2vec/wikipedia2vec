# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

import logging
import re
from icu import Locale, BreakIterator

from .sentence cimport Sentence


cdef class ICUSentenceDetector:
    cdef _locale
    cdef _breaker

    def __init__(self, locale):
        self._locale = locale
        self._breaker = BreakIterator.createSentenceInstance(Locale(locale))

    cpdef list detect_sentences(self, unicode text):
        self._breaker.setText(text)

        ret = []
        start = self._breaker.first()
        for end in self._breaker:
            ret.append(Sentence(text[start:end], start, end))
            start = end

        return ret

    def __reduce__(self):
        return (self.__class__, (self._locale,))
