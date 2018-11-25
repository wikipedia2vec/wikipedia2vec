# -*- coding: utf-8 -*-
# License: Apache License 2.0

cimport numpy as np
from libc.stdint cimport int32_t


cdef class Item:
    cdef readonly int32_t index
    cdef readonly int32_t count
    cdef readonly int32_t doc_count


cdef class Word(Item):
    cdef readonly unicode text


cdef class Entity(Item):
    cdef readonly unicode title


cdef class Dictionary:
    cdef readonly unicode uuid
    cdef readonly unicode language
    cdef readonly bint lowercase
    cdef readonly int32_t min_paragraph_len
    cdef readonly dict build_params
    cdef _word_dict
    cdef _entity_dict
    cdef _redirect_dict
    cdef readonly const int32_t [:, :] _word_stats
    cdef readonly const int32_t [:, :] _entity_stats
    cdef int32_t _entity_offset

    cpdef get_word(self, unicode, default=?)
    cpdef get_entity(self, unicode, bint resolve_redirect=?, default=?)
    cpdef int32_t get_word_index(self, unicode)
    cpdef int32_t get_entity_index(self, unicode, bint resolve_redirect=?)
    cpdef Item get_item_by_index(self, int32_t)
    cpdef Word get_word_by_index(self, int32_t)
    cpdef Entity get_entity_by_index(self, int32_t)
