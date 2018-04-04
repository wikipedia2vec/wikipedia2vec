# -*- coding: utf-8 -*-

from libc.stdint cimport uint32_t


cdef class PhraseDictionary:
    cdef public phrase_trie
    cdef bint _lowercase
    cdef dict _build_params

    cpdef list keys(self)
    cpdef list prefix_search(self, unicode, uint32_t start=?, uint32_t max_len=?)
