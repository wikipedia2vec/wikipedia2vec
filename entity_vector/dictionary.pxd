# -*- coding: utf-8 -*-

cimport numpy as np

cdef class Item:
    cdef public unsigned int index
    cdef public unsigned int count
    cdef public unsigned int doc_count


cdef class Word(Item):
    cdef public unicode text


cdef class Entity(Item):
    cdef public unicode title


cdef class Dictionary:
    cdef public str _id
    cdef public _dict
    cdef public _redirect_dict
    cdef public np.ndarray _ordered_keys
    cdef public int _size
    cdef public int _total_docs

    cpdef Word get_word(self, unicode)
    cpdef Entity get_entity(self, unicode, bint resolve_redirect=?)
    cpdef list get_bow_vector(self, object, bint tfidf=?, bint normalize=?)
    cdef inline Item _create_item_from_key(self, unicode)
    cdef inline unicode _create_entity_key(self, unicode)
    cdef inline unicode _get_title_from_key(self, unicode)
