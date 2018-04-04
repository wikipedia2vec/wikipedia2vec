# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import re

from wikipedia2vec.phrase cimport PhraseDictionary
from .base_tokenizer cimport BaseTokenizer


cdef class RegexpTokenizer(BaseTokenizer):
    cdef _rule

    def __init__(self, PhraseDictionary phrase_dict=None, rule=r'[\w\d]+'):
        super(RegexpTokenizer, self).__init__(phrase_dict)

        self._rule = re.compile(rule, re.UNICODE)

    cdef list _span_tokenize(self, unicode text):
        return [obj.span() for obj in self._rule.finditer(text)]
