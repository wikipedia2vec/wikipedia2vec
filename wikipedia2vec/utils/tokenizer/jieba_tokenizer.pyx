# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import jieba
import logging
import six

from .base_tokenizer cimport BaseTokenizer

jieba.setLogLevel(logging.WARN)


cdef class JiebaTokenizer(BaseTokenizer):
    cdef list _span_tokenize(self, unicode text):
        return [(start, end) for (_, start, end) in jieba.tokenize(text)]
