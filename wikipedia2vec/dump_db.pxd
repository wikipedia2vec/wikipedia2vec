# -*- coding: utf-8 -*-

from libc.stdint cimport int32_t

cdef class DumpDB:
    cdef _env
    cdef _meta_db
    cdef _page_db
    cdef _redirect_db

    cpdef list get_paragraphs(self, unicode)
    cdef list _deserialize_paragraphs(self, bytes)


cdef class Paragraph:
    cdef readonly unicode text
    cdef readonly list wiki_links


cdef class WikiLink:
    cdef readonly unicode title
    cdef readonly unicode text
    cdef readonly int32_t start
    cdef readonly int32_t end
