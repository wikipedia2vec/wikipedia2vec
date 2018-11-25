# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import re

from .base_tokenizer cimport BaseTokenizer


cdef class RegexpTokenizer(BaseTokenizer):
    cdef _rule
    cdef _rule_obj

    def __init__(self, rule=r'[\w\d]+'):
        self._rule = rule

        self._rule_obj = re.compile(rule, re.UNICODE)

    cdef list _span_tokenize(self, unicode text):
        return [obj.span() for obj in self._rule_obj.finditer(text)]

    def __reduce__(self):
        return (self.__class__, (self._rule,))
