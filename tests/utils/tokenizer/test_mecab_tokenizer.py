# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals
import unittest

from wikipedia2vec.utils.tokenizer.token import Token
from wikipedia2vec.utils.tokenizer.mecab_tokenizer import MeCabTokenizer

from nose.tools import *


class TestMeCabTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = MeCabTokenizer()

    def test_tokenize(self):
        text = '東京は日本の首都です'
        tokens = self._tokenizer.tokenize(text)

        ok_(all([isinstance(t, Token) for t in tokens]))
        eq_(['東京', 'は', '日本', 'の', '首都', 'です'], [t.text for t in tokens])
        eq_([(0, 2), (2, 3), (3, 5), (5, 6), (6, 8), (8, 10)], [t.span for t in tokens])
