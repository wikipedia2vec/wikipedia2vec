# -*- coding: utf-8 -*-

import pkg_resources
import cPickle as pickle
import unittest

from entity_vector.wiki_dump_reader import WikiDumpReader
from entity_vector.wiki_page import Word, WikiLink, WikiPage

from nose.tools import *


class TestWord(unittest.TestCase):
    def test_pickle(self):
        word1 = Word(u'text')
        word2 = pickle.loads(pickle.dumps(word1))

        ok_(isinstance(word2, Word))
        eq_(word1.text, word2.text)


class TestWikiLink(unittest.TestCase):
    def test_pickle(self):
        link1 = WikiLink(u'title', u'text', [Word(u'word1'), Word(u'word2')])
        link2 = pickle.loads(pickle.dumps(link1))

        ok_(isinstance(link2, WikiLink))
        eq_(link1.text, link2.text)
        eq_(link1.title, link2.title)
        eq_([w.text for w in link1.words], [w.text for w in link2.words])


class TestWikiPage(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename(
            __name__, './test_data/enwiki-pages-articles-sample.xml.bz2'
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

    def test_extract_paragraphs(self):
        wiki_text = u"""Barack Obama is a president of the United States"""
        wiki_page = WikiPage(u'TITLE', 'en', wiki_text)
        paragraphs = list(wiki_page.extract_paragraphs(min_paragraph_len=0))
        for (n, w) in enumerate(wiki_text.split()):
            eq_(w, paragraphs[0][n].text)
            ok_(isinstance(paragraphs[0][n], Word))

        wiki_text = u'[[Barack Obama|Obama]] is a president of the [[United States]].'
        wiki_page = WikiPage(u'TITLE', 'en', wiki_text)
        paragraphs = list(wiki_page.extract_paragraphs(min_paragraph_len=0))
        eq_(u'Obama', paragraphs[0][0].text)
        eq_(u'Barack Obama', paragraphs[0][0].title)
        eq_(u'Obama', paragraphs[0][0].words[0].text)
        ok_(isinstance(paragraphs[0][0], WikiLink))

        eq_(u'United States', paragraphs[0][-2].text)
        eq_(u'United States', paragraphs[0][-2].title)
        eq_(u'United', paragraphs[0][-2].words[0].text)
        eq_(u'States', paragraphs[0][-2].words[1].text)

    def test_extract_paragraphs_with_styled_text(self):
        wiki_text = u"""'''Barack Obama''' is a ''president'' of the '''United States'''"""
        wiki_page = WikiPage(u'TITLE', 'en', wiki_text)
        paragraphs = list(wiki_page.extract_paragraphs(min_paragraph_len=0))
        for (n, w) in enumerate(wiki_text.replace("'", '').split()):
            eq_(w, paragraphs[0][n].text)
            ok_(isinstance(paragraphs[0][n], Word))

    def test_extract_paragraphs_min_paragraph_len(self):
        wiki_text = u"Barack Obama is a president of the United States"
        wiki_page = WikiPage(u'TITLE', 'en', wiki_text)
        paragraphs = list(wiki_page.extract_paragraphs(min_paragraph_len=9))
        eq_(9, len(paragraphs[0]))
        paragraphs = list(wiki_page.extract_paragraphs(min_paragraph_len=10))
        eq_([], paragraphs)

    def test_extract_paragraphs_with_redirect_entry(self):
        eq_([], list(self.pages[0].extract_paragraphs()))
