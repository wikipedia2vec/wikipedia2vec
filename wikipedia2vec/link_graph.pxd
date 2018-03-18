# -*- coding: utf-8 -*-

cimport numpy as np

from .dictionary cimport Dictionary

cdef class LinkGraph:
    cdef Dictionary _dictionary
    cdef np.ndarray _indices
    cdef np.ndarray _indptr
    cdef dict _build_params
    cdef int _offset

    cpdef bint has_edge(self, object, object)
    cpdef list neighbors(self, object)
    cpdef np.ndarray neighbor_indices(self, object)
    cpdef list random_walk(self, object, int length=?, bint return_indices=?)
