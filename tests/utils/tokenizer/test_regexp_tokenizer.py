# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import unittest
from marisa_trie import Trie

from wikipedia2vec.phrase import PhraseDictionary
from wikipedia2vec.utils.tokenizer.token import Token
from wikipedia2vec.utils.tokenizer.regexp_tokenizer import RegexpTokenizer

from nose.tools import *


class TestRegexpTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = RegexpTokenizer()
        phrase_dict = PhraseDictionary(Trie(['New York City', 'New York', 'United States']), False, {})
        self._phrase_tokenizer = RegexpTokenizer(phrase_dict)

    def test_tokenize(self):
        text = 'Tokyo is the capital of Japan'
        tokens = self._tokenizer.tokenize(text)

        ok_(all([isinstance(t, Token) for t in tokens]))

        eq_(['Tokyo', 'is', 'the', 'capital', 'of', 'Japan'], [t.text for t in tokens])
        eq_([(0, 5), (6, 8), (9, 12), (13, 20), (21, 23), (24, 29)], [t.span for t in tokens])

    def test_phrase_tokenize(self):
        text = 'New York City is the capital city of the United States'
        tokens = self._phrase_tokenizer.tokenize(text)

        ok_(all([isinstance(t, Token) for t in tokens]))

        eq_(['New York City', 'is', 'the', 'capital', 'city', 'of', 'the', 'United States'],
            [t.text for t in tokens])
        eq_([(0, 13), (14, 16), (17, 20), (21, 28), (29, 33), (34, 36), (37, 40), (41, 54)],
            [t.span for t in tokens])
