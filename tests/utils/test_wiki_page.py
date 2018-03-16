# -*- coding: utf-8 -*-

from __future__ import absolute_import
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
        eq_(u'Computer accessibility', self.pages[0].redirect)

    def test_page_properties(self):
        page = self.pages[1]
        eq_('Computer accessibility', page.title)
        eq_('en', page.language)
        eq_(24949, len(page.wiki_text))

    def test_pickle(self):
        page1 = self.pages[1]
        page2 = pickle.loads(pickle.dumps(page1))

        eq_(page1.title, page2.title)
        eq_(page1.language, page2.language)
        eq_(page1.wiki_text, page2.wiki_text)
