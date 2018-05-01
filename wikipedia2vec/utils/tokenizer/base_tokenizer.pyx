# -*- coding: utf-8 -*-
# cython: profile=False

from .token cimport Token


cdef class BaseTokenizer:
    cpdef list tokenize(self, unicode text):
        return [Token(text[start:end], start, end) for (start, end) in self._span_tokenize(text)]

    cdef list _span_tokenize(self, unicode text):
        raise NotImplementedError()
