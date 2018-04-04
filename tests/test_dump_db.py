# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import unittest
import six.moves.cPickle as pickle

from wikipedia2vec.dump_db import Paragraph, WikiLink

from . import get_dump_db

from nose.tools import *


class TestDumpDB(unittest.TestCase):
    def test_paragraph(self):
        wiki_link = WikiLink('WikiTitle', 'link text', 0, 3)
        paragraph = Paragraph('paragraph text', [wiki_link])
        eq_(paragraph.text, 'paragraph text')
        eq_(paragraph.wiki_links, [wiki_link])

        paragraph2 = pickle.loads(pickle.dumps(paragraph))
        eq_(paragraph2.text, 'paragraph text')
        eq_(paragraph2.wiki_links[0].title, wiki_link.title)
        eq_(paragraph2.wiki_links[0].text, wiki_link.text)
        eq_(paragraph2.wiki_links[0].span, wiki_link.span)

    def test_wiki_link(self):
        wiki_link = WikiLink('WikiTitle', 'link text', 0, 3)
        eq_(wiki_link.title, 'WikiTitle')
        eq_(wiki_link.text, 'link text')
        eq_(wiki_link.start, 0)
        eq_(wiki_link.end, 3)
        eq_(wiki_link.span, (0, 3))

        wiki_link2 = pickle.loads(pickle.dumps(wiki_link))
        eq_(wiki_link2.title, wiki_link.title)
        eq_(wiki_link2.text, wiki_link.text)
        eq_(wiki_link2.start, wiki_link.start)
        eq_(wiki_link2.end, wiki_link.end)
        eq_(wiki_link2.span, wiki_link.span)

    def test_dump_db(self):
        db = get_dump_db()
        ok_(db.dump_file.endswith('enwiki-pages-articles-sample.xml.bz2'))
        eq_('en', db.language)
        eq_(2, db.page_size())
        eq_(['Accessibility', 'Computer accessibility'], list(db.titles()))
        eq_([('AccessibleComputing', 'Computer accessibility')], list(db.redirects()))

        paragraphs = db.get_paragraphs('Computer accessibility')
        paragraph = paragraphs[0]
        ok_(isinstance(paragraph, Paragraph))
        eq_('In  human–computer interaction', paragraph.text[:30])

        wiki_link = paragraph.wiki_links[0]
        ok_(isinstance(wiki_link, WikiLink))
        eq_('Human–computer interaction', wiki_link.title)
        eq_('human–computer interaction', wiki_link.text)
        eq_((3, 30), wiki_link.span)

    @raises(KeyError)
    def test_dump_db_no_key(self):
        get_dump_db().get_paragraphs('foo')
