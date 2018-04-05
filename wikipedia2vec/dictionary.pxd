# -*- coding: utf-8 -*-

cimport numpy as np
from libc.stdint cimport int32_t

from .phrase cimport PhraseDictionary
from .utils.tokenizer.base_tokenizer cimport BaseTokenizer


cdef class Item:
    cdef readonly int32_t index
    cdef readonly int32_t count
    cdef readonly int32_t doc_count


cdef class Word(Item):
    cdef readonly unicode text


cdef class Entity(Item):
    cdef readonly unicode title


cdef class Dictionary:
    cdef _word_dict
    cdef _entity_dict
    cdef _redirect_dict
    cdef PhraseDictionary _phrase_dict
    cdef const int32_t [:, :] _word_stats
    cdef const int32_t [:, :] _entity_stats
    cdef unicode _language
    cdef bint _lowercase
    cdef dict _build_params
    cdef int32_t _entity_offset

    cpdef get_word(self, unicode, default=?)
    cpdef get_entity(self, unicode, bint resolve_redirect=?, default=?)
    cpdef int32_t get_word_index(self, unicode)
    cpdef int32_t get_entity_index(self, unicode, bint resolve_redirect=?)
    cpdef Item get_item_by_index(self, int32_t)
    cpdef Word get_word_by_index(self, int32_t)
    cpdef Entity get_entity_by_index(self, int32_t)
    cpdef BaseTokenizer get_tokenizer(self)
