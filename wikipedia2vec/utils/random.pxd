# -*- coding: utf-8 -*-
# License: Apache License 2.0

from libc.stdint cimport int32_t, uint32_t

cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937:
        mt19937()
        mt19937(uint32_t)
        uint32_t operator()() nogil

    cdef cppclass uniform_int_distribution[T]:
        uniform_int_distribution()
        T operator()(mt19937) nogil


cpdef seed(uint32_t seed)
cpdef int32_t randint()
cdef int32_t randint_c() nogil
