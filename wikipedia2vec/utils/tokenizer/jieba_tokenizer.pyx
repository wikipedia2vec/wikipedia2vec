# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import jieba
import logging
import six

from .token cimport Token

jieba.setLogLevel(logging.WARN)


cdef class JiebaTokenizer:
    cpdef list tokenize(self, unicode text):
        return [Token(w, (s, e)) for (w, s, e) in jieba.tokenize(text)]
