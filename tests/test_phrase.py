# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import pkg_resources
import unittest

from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec.phrase import PhraseDictionary

from nose.tools import *


class TestPhraseDictionary(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename(
            __name__, 'test_data/enwiki-pages-articles-sample.xml.bz2'
        )
        self.dump_reader = WikiDumpReader(sample_dump_file)

    def test_phrase_dictionary(self):
        phrase_dic = PhraseDictionary.build(self.dump_reader, min_link_count=0, min_link_prob=0.1,
                                            lowercase=True, max_len=4, pool_size=1, chunk_size=1)
        ok_(len(phrase_dic) > 0)
        ok_('computer system' in phrase_dic)
        eq_(phrase_dic.prefix_search('web accessibility initiative'),
            ['web accessibility initiative', 'web accessibility'])
        eq_(sorted(phrase_dic.keys()), sorted(list(phrase_dic)))
