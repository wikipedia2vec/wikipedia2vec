# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import MeCab
import six
from libc.stdint cimport int32_t

from wikipedia2vec.phrase cimport PhraseDictionary
from .base_tokenizer cimport BaseTokenizer


cdef class MeCabTokenizer(BaseTokenizer):
    cdef _tagger

    def __init__(self, PhraseDictionary phrase_dict=None):
        super(MeCabTokenizer, self).__init__(phrase_dict)

        if six.PY2:
            self._tagger = MeCab.Tagger(''.encode('utf-8'))
        else:
            self._tagger = MeCab.Tagger('')
            self._tagger.parse('')

    cdef list _span_tokenize(self, unicode text):
        cdef int32_t cur, space_length, start, end
        cdef bytes text_utf8
        cdef unicode word
        cdef list ret

        if six.PY2:
            text_utf8 = text.encode('utf-8')
            node = self._tagger.parseToNode(text_utf8)
        else:
            node = self._tagger.parseToNode(text)

        cur = 0
        ret = []

        while node:
            if node.stat not in (2, 3):
                if six.PY2:
                    word = node.surface.decode('utf-8')
                else:
                    word = node.surface
                space_length = node.rlength - node.length

                start = cur + space_length
                end = start + len(word)

                ret.append((start, end))

                cur += len(word) + space_length

            node = node.next

        return ret
