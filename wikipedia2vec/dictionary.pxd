# -*- coding: utf-8 -*-

cimport numpy as np
from libc.stdint cimport int32_t, uint32_t


cdef class Item:
    cdef int32_t _index
    cdef uint32_t _count
    cdef uint32_t _doc_count


cdef class Word(Item):
    cdef unicode _text


cdef class Entity(Item):
    cdef unicode _title


cdef class PrefixSearchable:
    cpdef list prefix_search(self, unicode, int32_t start=?)


cdef class Dictionary(PrefixSearchable):
    cdef _word_dict
    cdef _entity_dict
    cdef _redirect_dict
    cdef np.ndarray _word_stats
    cdef np.ndarray _entity_stats
    cdef int32_t _entity_offset
    cdef _lowercase
    cdef dict _build_params

    cpdef get_word(self, unicode, default=?)
    cpdef get_entity(self, unicode, bint resolve_redirect=?, default=?)
    cpdef int32_t get_word_index(self, unicode)
    cpdef int32_t get_entity_index(self, unicode, bint resolve_redirect=?)
    cpdef Item get_item_by_index(self, int32_t)
    cpdef Word get_word_by_index(self, int32_t)
    cpdef Entity get_entity_by_index(self, int32_t)
    cpdef list prefix_search(self, unicode, int32_t start=?)
