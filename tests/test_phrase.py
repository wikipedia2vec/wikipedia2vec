# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import unittest
from tempfile import NamedTemporaryFile

from wikipedia2vec.phrase import PhraseDictionary

from . import get_dump_db

from nose.tools import *


class TestPhraseDictionary(unittest.TestCase):
    def setUp(self):
        self.phrase_dic = PhraseDictionary.build(get_dump_db(), min_link_count=0, min_link_prob=0.1,
                                                 lowercase=True, max_phrase_len=3, pool_size=1,
                                                 chunk_size=1, progressbar=False)

    def test_properties(self):
        ok_(self.phrase_dic.lowercase)

    def test_methods(self):
        ok_(len(self.phrase_dic) > 0)
        ok_('computer system' in self.phrase_dic)
        ok_(isinstance(self.phrase_dic.keys(), list))
        eq_(len(self.phrase_dic.keys()), len(self.phrase_dic))
        eq_(sorted(self.phrase_dic.keys()), sorted(list(self.phrase_dic)))

    def test_prefix_search(self):
        eq_(self.phrase_dic.prefix_search('web accessibility initiative'),
            ['web accessibility initiative', 'web accessibility'])

    def test_save_load(self):
        eq_(self.phrase_dic.load(self.phrase_dic.serialize()).serialize(),
            self.phrase_dic.serialize())

        with NamedTemporaryFile() as f:
            self.phrase_dic.save(f.name)
            eq_(self.phrase_dic.load(f.name).serialize(), self.phrase_dic.serialize())
