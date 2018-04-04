# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import unittest
from marisa_trie import Trie

from wikipedia2vec.phrase import PhraseDictionary
from wikipedia2vec.utils.tokenizer.token import Token
from wikipedia2vec.utils.tokenizer.mecab_tokenizer import MeCabTokenizer

from nose.tools import *


class TestMeCabTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = MeCabTokenizer()
        phrase_dict = PhraseDictionary(Trie(['充実野菜']), False, {})
        self._phrase_tokenizer = MeCabTokenizer(phrase_dict)

    def test_tokenize(self):
        text = '東京は日本の首都です'
        tokens = self._tokenizer.tokenize(text)

        ok_(all([isinstance(t, Token) for t in tokens]))

        eq_(['東京', 'は', '日本', 'の', '首都', 'です'], [t.text for t in tokens])
        eq_([(0, 2), (2, 3), (3, 5), (5, 6), (6, 8), (8, 10)], [t.span for t in tokens])

    def test_phrase_tokenize(self):
        text = '充実野菜は野菜ジュースです'
        tokens = self._phrase_tokenizer.tokenize(text)

        ok_(all([isinstance(t, Token) for t in tokens]))

        eq_(['充実野菜', 'は', '野菜', 'ジュース', 'です'], [t.text for t in tokens])
        eq_([(0, 4), (4, 5), (5, 7), (7, 11), (11, 13)], [t.span for t in tokens])
