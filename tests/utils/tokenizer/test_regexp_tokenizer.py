# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals
import unittest

from wikipedia2vec.utils.tokenizer.token import Token
from wikipedia2vec.utils.tokenizer.regexp_tokenizer import RegexpTokenizer

from nose.tools import *


class TestRegexpTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = RegexpTokenizer()

    def test_tokenize(self):
        text = 'Tokyo is the capital of Japan'
        tokens = self._tokenizer.tokenize(text)

        ok_(all([isinstance(t, Token) for t in tokens]))
        eq_(['Tokyo', 'is', 'the', 'capital', 'of', 'Japan'], [t.text for t in tokens])
        eq_([(0, 5), (6, 8), (9, 12), (13, 20), (21, 23), (24, 29)], [t.span for t in tokens])
