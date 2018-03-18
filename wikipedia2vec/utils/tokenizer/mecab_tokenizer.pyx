# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import six

from .token cimport Token


cdef class MeCabTokenizer:
    cdef object _tagger

    def __init__(self):
        import MeCab
        if six.PY2:
            self._tagger = MeCab.Tagger(''.encode('utf-8'))
        else:
            self._tagger = MeCab.Tagger('')
            self._tagger.parse('')

    cpdef list tokenize(self, unicode text):
        if six.PY2:
            text_utf8 = text.encode('utf-8')
            node = self._tagger.parseToNode(text_utf8)
        else:
            node = self._tagger.parseToNode(text)
        cur = 0
        tokens = []

        while node:
            if node.stat not in (2, 3):
                if six.PY2:
                    word = node.surface.decode('utf-8')
                else:
                    word = node.surface
                space_length = node.rlength - node.length

                begin = cur + space_length
                end = begin + len(word)

                tokens.append(Token(word, (begin, end)))

                cur += len(word) + space_length

            node = node.next

        return tokens
