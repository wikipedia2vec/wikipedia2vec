# -*- coding: utf-8 -*-

from entity_vector.utils.tokenizer.token cimport Token


cdef class MeCabTokenizer:
    cdef object _tagger

    def __init__(self):
        import MeCab
        self._tagger = MeCab.Tagger('')

    cpdef list tokenize(self, unicode text):
        node = self._tagger.parseToNode(text.encode('utf-8'))
        cur = 0
        tokens = []

        while node:
            if not node.stat in (2, 3):
                word = node.surface.decode('utf-8')
                space_length = node.rlength - node.length

                begin = cur + space_length
                end = begin + len(word)

                tokens.append(Token(word, (begin, end)))

                cur += len(word) + space_length

            node = node.next

        return tokens
