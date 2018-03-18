# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import pkg_resources
import unittest

from wikipedia2vec.extractor import Extractor, Paragraph, WikiLink
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

from nose.tools import *


class TestWikiPage(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename(
            __name__, 'test_data/enwiki-pages-articles-sample.xml.bz2'
        )
        self.dump_reader = WikiDumpReader(sample_dump_file)
        self.page = list(self.dump_reader)[1]

    def test_extractor(self):
        extractor = Extractor('en')
        paragraphs = extractor.extract_paragraphs(self.page)
        ok_(isinstance(paragraphs[0], Paragraph))
        eq_(paragraphs[0].words[:3], ['in', 'human', 'computer'])

        ok_(isinstance(paragraphs[0].wiki_links[0], WikiLink))
        eq_(paragraphs[0].wiki_links[0].title, 'Human\u2013computer interaction')
        eq_(paragraphs[0].wiki_links[0].text, 'human\u2013computer interaction')
        eq_(paragraphs[0].wiki_links[0].span, (1, 4))

        extractor = Extractor('en', lowercase=False)
        paragraphs = extractor.extract_paragraphs(self.page)
        eq_(paragraphs[0].words[:3], ['In', 'human', 'computer'])
