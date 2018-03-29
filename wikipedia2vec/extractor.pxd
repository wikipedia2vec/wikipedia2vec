# -*- coding: utf-8 -*-

cimport numpy as np
from libc.stdint cimport int32_t, uint32_t

from .dictionary cimport PrefixSearchable
from .utils.wiki_page cimport WikiPage


cdef class Paragraph:
    cdef unicode _text
    cdef list _words
    cdef list _wiki_links


cdef class WikiLink:
    cdef unicode _title
    cdef unicode _text
    cdef tuple _span


cdef class Extractor:
    cdef unicode _language
    cdef bint _lowercase
    cdef uint32_t _min_paragraph_len
    cdef _tokenizer
    cdef PrefixSearchable _dictionary

    cpdef list extract_paragraphs(self, WikiPage)
    cpdef list _extract_words(self, unicode)
    cpdef _parse_page(self, WikiPage)
    cdef inline unicode _normalize_title(self, unicode)
