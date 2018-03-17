# -*- coding: utf-8 -*-


cdef class WikiPage:
    cdef unicode _title
    cdef unicode _language
    cdef unicode _wiki_text

    cdef inline unicode _normalize_title(self, unicode)
