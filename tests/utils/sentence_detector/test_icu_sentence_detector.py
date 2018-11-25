# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals
import unittest

from wikipedia2vec.utils.sentence_detector.sentence import Sentence
from wikipedia2vec.utils.sentence_detector.icu_sentence_detector import ICUSentenceDetector

from nose.tools import *


class TestICUSentenceDetector(unittest.TestCase):
    def setUp(self):
        self._detector = ICUSentenceDetector('en')

    def test_detect_sentences(self):
        text = 'Wikipedia is an encyclopedia based on a model of openly editable content. It is the largest general reference work on the Internet.'
        sents = self._detector.detect_sentences(text)

        ok_(all([isinstance(s, Sentence) for s in sents]))
        eq_('Wikipedia is an encyclopedia based on a model of openly editable content. ', sents[0].text)
        eq_('It is the largest general reference work on the Internet.', sents[1].text)
        eq_([(0, 74), (74, 131)], [s.span for s in sents])
