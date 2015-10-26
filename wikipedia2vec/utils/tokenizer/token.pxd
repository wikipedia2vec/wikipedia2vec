# -*- coding: utf-8 -*-


cdef class Token:
    cdef public unicode text
    cdef public tuple span
