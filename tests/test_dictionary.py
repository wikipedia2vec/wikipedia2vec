# -*- coding: utf-8 -*-

import pkg_resources
import unittest
import cPickle as pickle
import numpy as np
from contextlib import closing
from cStringIO import StringIO

from entity_vector.wiki_dump_reader import WikiDumpReader
from entity_vector.dictionary import Dictionary, Item, Word, Entity

from nose.tools import *
import nose.tools


class TestWord(unittest.TestCase):
    def test_pickle(self):
        word = Word(u'text', 1, 10, 5)
        word2 = pickle.loads(pickle.dumps(word))
        eq_(u'text', word2.text)
        eq_(1, word2.index)
        eq_(10, word2.count)
        eq_(5, word2.doc_count)


class TestEntity(unittest.TestCase):
    def test_pickle(self):
        entity = Entity(u'title', 1, 10, 5)
        entity2 = pickle.loads(pickle.dumps(entity))
        eq_(u'title', entity2.title)
        eq_(1, entity2.index)
        eq_(10, entity2.count)
        eq_(5, entity2.doc_count)


class TestDictionary(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename(
            __name__, './test_data/enwiki-pages-articles-sample.xml.bz2'
        )
        dump_reader = WikiDumpReader(sample_dump_file)
        self.dictionary = Dictionary.build(
            dump_reader, min_word_count=1, min_entity_count=1, parallel=False
        )

    def test_ordered_keys(self):
        ordered_keys = self.dictionary.ordered_keys
        eq_(len(self.dictionary), len(ordered_keys))
        ok_(isinstance(ordered_keys, np.ndarray))

        for k in ordered_keys:
            ok_(isinstance(k, unicode))

    def test_get_word(self):
        word = self.dictionary.get_word(u'a')
        ok_(isinstance(word, Word))
        eq_(u'a', word.text)
        eq_(171, word.count)
        eq_(2, word.doc_count)
        eq_(1070, word.index)

    def test_get_entity(self):
        entity = self.dictionary.get_entity(u'Internet')
        ok_(isinstance(entity, Entity))
        eq_(u'Internet', entity.title)
        eq_(1, entity.count)
        eq_(1, entity.doc_count)
        eq_(1908, entity.index)

    def test_get_bow_vector(self):
        s = [u'entity', u'essential', u'emotional']
        expected = [
            (self.dictionary.get_word(u'entity'), 0.7071068286895752),
            (self.dictionary.get_word(u'essential'), 0.7071068286895752)
        ]
        ret = sorted(self.dictionary.get_bow_vector(s), key=lambda o: o[0].text)
        for (item1, item2) in zip(expected, ret):
            eq_(item1[0].text, item2[0].text)
            nose.tools.assert_almost_equals(item1[1], item2[1])

        expected = [
            (self.dictionary.get_word(u'entity'), 1),
            (self.dictionary.get_word(u'essential'), 1)
        ]
        ret = sorted(
            self.dictionary.get_bow_vector(s, tfidf=False, normalize=False),
            key=lambda o: o[0].text
        )
        for (item1, item2) in zip(expected, ret):
            eq_(item1[0].text, item2[0].text)
            eq_(item1[1], item2[1])

    def test_words_iterator(self):
        for (n, word) in enumerate(self.dictionary.words()):
            ok_(isinstance(word, Word))
        eq_(1901, n)

    def test_entities_iterator(self):
        for (n, entity) in enumerate(self.dictionary.entities()):
            ok_(isinstance(entity, Entity))
        eq_(161, n)

    def test_id(self):
        ok_(isinstance(self.dictionary.id, str))

    def test_total_docs(self):
        eq_(2, self.dictionary.total_docs)

    def test_size(self):
        eq_(2064, len(self.dictionary))

    def test_contains(self):
        ok_(u'a' in self.dictionary)

    def test_iter(self):
        for (n, item) in enumerate(self.dictionary):
            ok_(isinstance(item, Item))

        eq_(len(self.dictionary) - 1, n)

    def test_getitem_by_index(self):
        word = self.dictionary[0]
        ok_(isinstance(word, Word))
        eq_(u'limited', word.text)
        eq_(5, word.count)
        eq_(0, word.index)

    def test_getitem_by_word(self):
        word = self.dictionary[u'a']
        ok_(isinstance(word, Word))
        eq_(u'a', word.text)
        eq_(171, word.count)
        eq_(1070, word.index)

    def test_getitem_by_entity_title(self):
        entity = self.dictionary[u'Internet']
        ok_(isinstance(entity, Entity))
        eq_(u'Internet', entity.title)
        eq_(1, entity.count)
        eq_(1908, entity.index)

    def test_pickle(self):
        dictionary2 = pickle.loads(pickle.dumps(self.dictionary))
        eq_(len(self.dictionary), len(dictionary2))
        for (item1, item2) in zip(self.dictionary, dictionary2):
            eq_(pickle.dumps(item1), pickle.dumps(item2))

    def test_save_and_load(self):
        with closing(StringIO()) as f:
            self.dictionary.save(f)
            f.seek(0)
            dictionary2 = Dictionary.load(f)

        eq_(len(self.dictionary), len(dictionary2))
        for (item1, item2) in zip(self.dictionary, dictionary2):
            eq_(pickle.dumps(item1), pickle.dumps(item2))
