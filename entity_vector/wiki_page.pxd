# -*- coding: utf-8 -*-


cdef class Word:
    cdef public unicode text


cdef class WikiLink:
    cdef public unicode title
    cdef public unicode text
    cdef public list words


cdef class WikiPage:
    cdef public unicode title
    cdef public str language
    cdef public unicode wiki_text
    cdef inline unicode _normalize_title(self, unicode)
