cdef class Sentence:
    cdef readonly str text
    cdef readonly int start
    cdef readonly int end