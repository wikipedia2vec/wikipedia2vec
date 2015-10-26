# -*- coding: utf-8 -*-

import re

from token cimport Token


cdef class RegexpTokenizer:
    cdef _rule

    # def __init__(self, rule='(((?![\d])\w)+)'):
    def __init__(self, rule=ur'[\w\d]+'):
        self._rule = re.compile(rule, re.UNICODE)

    cpdef list tokenize(self, unicode text):
        return [Token(text[o.start():o.end()], o.span())
                for o in self._rule.finditer(text)]
