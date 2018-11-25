# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals
import pickle
import unittest

from wikipedia2vec.utils.tokenizer.token import Token

from nose.tools import *


class TestToken(unittest.TestCase):
    def test_text_property(self):
        token = Token('text', 1, 3)
        eq_('text', token.text)

    def test_start_property(self):
        token = Token('text', 1, 3)
        eq_(1, token.start)

    def test_end_property(self):
        token = Token('text', 1, 3)
        eq_(3, token.end)

    def test_span_property(self):
        token = Token('text', 1, 3)
        eq_((1, 3), token.span)

    def test_pickle(self):
        token = pickle.loads(pickle.dumps(Token('text', 1, 3)))
        eq_('text', token.text)
        eq_(1, token.start)
        eq_(3, token.end)
