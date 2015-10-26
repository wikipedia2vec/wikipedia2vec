# -*- coding: utf-8 -*-

import unittest

from entity_vector.utils.tokenizer.token import Token
from entity_vector.utils.tokenizer.opennlp import OpenNLPTokenizer, OpenNLPSentenceDetector

from nose.tools import *


class TestOpenNLPSentenceDetector(unittest.TestCase):
    def setUp(self):
        self._detector = OpenNLPSentenceDetector()

    def test_sent_pos_detect(self):
        text = u'Tokyo is the capital of Japan. It is the seat of the Japanese government.'

        eq_([(0, 30), (31, 73)], self._detector.sent_pos_detect(text))


class TestOpenNLPTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = OpenNLPTokenizer()

    def test_tokenize(self):
        text = u'Tokyo is the capital of Japan.'
        tokens = self._tokenizer.tokenize(text)

        ok_(all([isinstance(t, Token) for t in tokens]))

        eq_([(0, 5), (6, 8), (9, 12), (13, 20), (21, 23), (24, 29), (29, 30)],
            [t.span for t in tokens])

        eq_([u'Tokyo', u'is', u'the', u'capital', u'of', u'Japan', u'.'],
            [t.text for t in tokens])
