# -*- coding: utf-8 -*-
# cython: profile=False

import cython
from cython cimport view

from cython.operator cimport dereference as deref, preincrement as inc
from libc.stdint cimport uint32_t
from libc.string cimport memset

from .token cimport Token


cdef class BaseTokenizer:
    def __init__(self, PhraseDictionary phrase_dict=None):
        self._phrase_dict = phrase_dict

    cpdef list tokenize(self, unicode text):
        cdef uint32_t cur, start, end, phrase_end
        cdef bint matched
        cdef unicode target_text, prefix
        cdef list ret, spans
        cdef tuple span
        cdef char [:] end_markers

        spans = self._span_tokenize(text)
        ret = []

        if self._phrase_dict is None:
            for (start, end) in spans:
                ret.append(Token(text[start:end], start, end))

        else:
            if self._phrase_dict._lowercase:
                target_text = text.lower()
            else:
                target_text = text

            end_markers = view.array(shape=(len(text) + 1,), itemsize=cython.sizeof(char),
                                     format='c')
            memset(&end_markers[0], 0, (len(text) + 1) * cython.sizeof(char))

            for (_, end) in spans:
                end_markers[end] = 1

            cur = 0
            for (start, end) in spans:
                if cur > start:
                    continue

                matched = False
                for prefix in self._phrase_dict.prefix_search(target_text, start=start):
                    phrase_end = start + len(prefix)
                    if end_markers[phrase_end] == 1:
                        ret.append(Token(text[start:phrase_end], start, phrase_end))
                        cur = phrase_end
                        matched = True
                        break

                if not matched:
                    ret.append(Token(text[start:end], start, end))

        return ret

    cdef list _span_tokenize(self, unicode text):
        raise NotImplementedError()
