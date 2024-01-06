cdef class WikiLink:
    cdef readonly str title
    cdef readonly str text
    cdef readonly int start
    cdef readonly int end

cdef class Paragraph:
    cdef readonly str text
    cdef readonly list[WikiLink] wiki_links
    cdef readonly bint abstract
