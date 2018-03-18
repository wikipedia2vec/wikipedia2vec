# -*- coding: utf-8 -*-

from .dictionary cimport PrefixSearchable


cdef class PhraseDictionary(PrefixSearchable):
    cdef _phrase_dict
    cdef bint _lowercase
    cdef dict _build_params

    cpdef list keys(self)
    cpdef list prefix_search(self, unicode, int start=?)
