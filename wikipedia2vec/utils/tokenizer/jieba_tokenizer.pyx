# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import jieba
import logging
import re
import six

from .base_tokenizer cimport BaseTokenizer

jieba.setLogLevel(logging.WARN)


cdef class JiebaTokenizer(BaseTokenizer):
    cdef _rule

    def __init__(self):
        self._rule = re.compile(r'^\s*$')

    cdef list _span_tokenize(self, unicode text):
        return [(start, end) for (word, start, end) in jieba.tokenize(text)
                if not self._rule.match(word)]

    def __reduce__(self):
        return (self.__class__, tuple())
