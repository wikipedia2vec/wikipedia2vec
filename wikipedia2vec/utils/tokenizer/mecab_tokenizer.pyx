# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .token cimport Token


cdef class MeCabTokenizer:
    cdef object _tagger

    def __init__(self):
        import MeCab
        self._tagger = MeCab.Tagger('')

    cpdef list tokenize(self, unicode text):
        text_utf8 = text.encode('utf-8')
        node = self._tagger.parseToNode(text_utf8)
        cur = 0
        tokens = []

        while node:
            if node.stat not in (2, 3):
                word = node.surface.decode('utf-8')
                space_length = node.rlength - node.length

                begin = cur + space_length
                end = begin + len(word)

                tokens.append(Token(word, (begin, end)))

                cur += len(word) + space_length

            node = node.next

        return tokens
