# -*- coding: utf-8 -*-

cimport numpy as np

from dictionary cimport Dictionary


cdef class ClickGraph:
    cdef Dictionary _dictionary
    cdef np.ndarray _ind_ptr
    cdef np.ndarray _keys
    cdef np.ndarray _probs
    cdef np.ndarray _aliases
    cdef dict _build_params
    cdef int _offset

    cpdef bint has_edge(self, object, object)

    cpdef list neighbors(self, object)
    cpdef np.ndarray neighbor_indices(self, object)
    cpdef list random_walk(self, item, int length=?, bint return_indices=?)
