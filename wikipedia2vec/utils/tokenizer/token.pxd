# -*- coding: utf-8 -*-

from libc.stdint cimport uint32_t


cdef class Token:
    cdef readonly unicode text
    cdef readonly uint32_t start
    cdef readonly uint32_t end
