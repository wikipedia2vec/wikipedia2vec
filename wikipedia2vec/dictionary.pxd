# -*- coding: utf-8 -*-

cimport numpy as np


cdef class Item:
    cdef int _index
    cdef int _count
    cdef int _doc_count


cdef class Word(Item):
    cdef unicode _text


cdef class Entity(Item):
    cdef unicode _title


cdef class PrefixSearchable:
    cpdef list prefix_search(self, unicode, int start=?)


cdef class Dictionary(PrefixSearchable):
    cdef _word_dict
    cdef _entity_dict
    cdef _redirect_dict
    cdef np.ndarray _word_stats
    cdef np.ndarray _entity_stats
    cdef int _entity_offset
    cdef bint _lowercase
    cdef dict _build_params

    cpdef get_word(self, unicode, default=?)
    cpdef get_entity(self, unicode, bint resolve_redirect=?, default=?)
    cpdef int get_word_index(self, unicode)
    cpdef int get_entity_index(self, unicode, bint resolve_redirect=?)
    cpdef Item get_item_by_index(self, int)
    cpdef Word get_word_by_index(self, int)
    cpdef Entity get_entity_by_index(self, int)
    cpdef list prefix_search(self, unicode, int start=?)
