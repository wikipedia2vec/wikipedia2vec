# -*- coding: utf-8 -*-

from libc.stdint cimport int32_t


cdef class Sentence:
    cdef readonly unicode text
    cdef readonly int32_t start
    cdef readonly int32_t end
