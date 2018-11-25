# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals
import pkg_resources
import unittest
import six.moves.cPickle as pickle

from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

from nose.tools import *


class TestWikiPage(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename(
            __name__, '../test_data/enwiki-pages-articles-sample.xml.bz2'
        )
        self.dump_reader = WikiDumpReader(sample_dump_file)
        self.pages = list(self.dump_reader)

    def test_is_redirect(self):
        ok_(self.pages[0].is_redirect)

    def test_redirect(self):
        eq_('Computer accessibility', self.pages[0].redirect)

    def test_title_property(self):
        eq_('Computer accessibility', self.pages[1].title)

    def test_language_property(self):
        eq_('en', self.pages[1].language)

    def test_wiki_text_property(self):
        eq_(24949, len(self.pages[1].wiki_text))

    def test_pickle(self):
        page1 = self.pages[1]
        page2 = pickle.loads(pickle.dumps(page1))

        eq_(page1.title, page2.title)
        eq_(page1.language, page2.language)
        eq_(page1.wiki_text, page2.wiki_text)
