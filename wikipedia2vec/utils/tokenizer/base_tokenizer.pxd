# -*- coding: utf-8 -*-

from libc.stdint cimport uint32_t

from wikipedia2vec.phrase cimport PhraseDictionary

cdef class BaseTokenizer:
    cdef readonly PhraseDictionary _phrase_dict
    cpdef list tokenize(self, unicode)
    cdef list _span_tokenize(self, unicode)
