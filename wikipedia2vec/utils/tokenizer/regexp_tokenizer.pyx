# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import re

from .token cimport Token


cdef class RegexpTokenizer:
    cdef _rule

    def __init__(self, rule=r'[\w\d]+'):
        self._rule = re.compile(rule, re.UNICODE)

    cpdef list tokenize(self, unicode text):
        return [Token(text[o.start():o.end()], o.span()) for o in self._rule.finditer(text)]
