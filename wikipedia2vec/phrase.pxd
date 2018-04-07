# -*- coding: utf-8 -*-

from libc.stdint cimport int32_t


cdef class PhraseDictionary:
    cdef public phrase_trie
    cdef readonly unicode uuid
    cdef readonly bint lowercase
    cdef readonly dict build_params

    cpdef list keys(self)
    cpdef list prefix_search(self, unicode, int32_t start=?, int32_t max_len=?)
