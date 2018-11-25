# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

cdef mt19937 _rng
cdef uniform_int_distribution[int32_t] _dist


cpdef seed(uint32_t n):
    global _rng
    _rng = mt19937(n)


cpdef int32_t randint():
    return randint_c()


cdef int32_t randint_c() nogil:
    return _dist(_rng)
