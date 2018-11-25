# -*- coding: utf-8 -*-
# License: Apache License 2.0

from libc.stdint cimport int32_t


cdef class Token:
    cdef readonly unicode text
    cdef readonly int32_t start
    cdef readonly int32_t end
