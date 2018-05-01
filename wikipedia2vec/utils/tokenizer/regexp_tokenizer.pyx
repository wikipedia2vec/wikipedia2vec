# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import re

from .base_tokenizer cimport BaseTokenizer


cdef class RegexpTokenizer(BaseTokenizer):
    cdef _rule

    def __init__(self, rule=r'[\w\d]+'):
        self._rule = re.compile(rule, re.UNICODE)

    cdef list _span_tokenize(self, unicode text):
        return [obj.span() for obj in self._rule.finditer(text)]
