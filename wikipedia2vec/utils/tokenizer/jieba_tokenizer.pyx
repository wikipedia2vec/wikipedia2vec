# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import jieba
import logging
import six

from .base_tokenizer cimport BaseTokenizer

jieba.setLogLevel(logging.WARN)


cdef class JiebaTokenizer(BaseTokenizer):
    cdef list _span_tokenize(self, unicode text):
        return [(start, end) for (_, start, end) in jieba.tokenize(text)]

    def __reduce__(self):
        return (self.__class__, tuple())
