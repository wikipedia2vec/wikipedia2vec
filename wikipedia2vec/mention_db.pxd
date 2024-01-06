cdef class Mention:
    cdef readonly str text
    cdef readonly int index
    cdef readonly int link_count
    cdef readonly int total_link_count
    cdef readonly int doc_count
    cdef readonly int start
    cdef readonly int end
    cdef _dictionary
