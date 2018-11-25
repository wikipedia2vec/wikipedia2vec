# -*- coding: utf-8 -*-
# License: Apache License 2.0


cdef class WikiPage:
    cdef readonly unicode title
    cdef readonly unicode language
    cdef readonly unicode wiki_text
    cdef readonly unicode redirect
