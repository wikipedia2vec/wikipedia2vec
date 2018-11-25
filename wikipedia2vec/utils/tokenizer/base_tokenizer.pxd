# -*- coding: utf-8 -*-
# License: Apache License 2.0


cdef class BaseTokenizer:
    cpdef list tokenize(self, unicode)
    cdef list _span_tokenize(self, unicode)
