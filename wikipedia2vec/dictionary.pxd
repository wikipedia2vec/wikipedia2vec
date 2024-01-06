cdef class Item:
    cdef readonly int index
    cdef readonly int count
    cdef readonly int doc_count


cdef class Word(Item):
    cdef readonly str text


cdef class Entity(Item):
    cdef readonly str title
