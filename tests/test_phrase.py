# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import six
import unittest
from tempfile import NamedTemporaryFile

from wikipedia2vec.phrase import PhraseDictionary

from . import get_dump_db

from nose.tools import *


class TestPhraseDictionary(unittest.TestCase):
    def setUp(self):
        self.phrase_dic = PhraseDictionary.build(get_dump_db(), min_link_count=1, min_link_prob=0.1,
                                                 lowercase=True, max_phrase_len=3, pool_size=1,
                                                 chunk_size=1, progressbar=False)

    def test_lowercase_property(self):
        ok_(self.phrase_dic.lowercase)

    def test_build_params_property(self):
        build_params = self.phrase_dic.build_params
        eq_(1, build_params['min_link_count'])
        eq_(0.1, build_params['min_link_prob'])
        ok_(build_params['dump_file'].endswith('enwiki-pages-articles-sample.xml.bz2'))
        ok_(isinstance(build_params['build_time'], float))
        ok_(build_params['build_time'] > 0)

    def test_len(self):
        eq_(116, len(self.phrase_dic))

    def test_iterator(self):
        phrases = list(self.phrase_dic)
        eq_(116, len(phrases))
        ok_(all(isinstance(phrase, six.text_type) for phrase in phrases))

    def test_contains(self):
        ok_('computer system' in self.phrase_dic)

    def test_keys(self):
        phrases = self.phrase_dic.keys()
        ok_(isinstance(phrases, list))
        eq_(116, len(phrases))
        ok_(all(isinstance(phrase, six.text_type) for phrase in phrases))

    def test_prefix_search(self):
        eq_(['web accessibility initiative', 'web accessibility'],
            self.phrase_dic.prefix_search('web accessibility initiative'))

    def test_save_load(self):
        eq_(self.phrase_dic.load(self.phrase_dic.serialize()).serialize(),
            self.phrase_dic.serialize())
        eq_(self.phrase_dic.keys(),
            self.phrase_dic.load(self.phrase_dic.serialize()).keys())

        with NamedTemporaryFile() as f:
            self.phrase_dic.save(f.name)
            eq_(self.phrase_dic.serialize(), self.phrase_dic.load(f.name).serialize())
            eq_(self.phrase_dic.keys(), self.phrase_dic.load(f.name).keys())
