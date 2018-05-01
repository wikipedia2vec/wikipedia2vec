# -*- coding: utf-8 -*-


cdef class BaseTokenizer:
    cpdef list tokenize(self, unicode)
    cdef list _span_tokenize(self, unicode)
