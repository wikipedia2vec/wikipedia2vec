# -*- coding: utf-8 -*-

import unittest

from entity_vector.utils.ner.stanford_ner import StanfordNER

from nose.tools import *


class TestStanfordNER(unittest.TestCase):
    def setUp(self):
        self._ner = StanfordNER()

    def test_extract(self):
        text = u'Washington is the capital of the United States'.split()

        ret = self._ner.extract(text)
        eq_(2, len(ret))
        eq_('LOCATION', ret[0].entity_type)
        eq_((0, 1), ret[0].span)
        eq_('LOCATION', ret[1].entity_type)
        eq_((6, 8), ret[1].span)

        text = u'Steve Jobs is the founder of Apple Inc.'.split()
        ret = self._ner.extract(text)
        eq_(2, len(ret))
        eq_('PERSON', ret[0].entity_type)
        eq_((0, 2), ret[0].span)
        eq_('ORGANIZATION', ret[1].entity_type)
        eq_((6, 8), ret[1].span)
