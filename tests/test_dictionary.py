# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import numpy as np
import pickle
import six
import unittest
from tempfile import NamedTemporaryFile

from wikipedia2vec.dictionary import Dictionary, Item, Word, Entity
from wikipedia2vec.phrase import PhraseDictionary
from wikipedia2vec.utils.tokenizer.base_tokenizer import BaseTokenizer

from . import get_dump_db

from nose.tools import *


class TestWord(unittest.TestCase):
    def test_text_property(self):
        word = Word('text', 1000, 100, 10)
        eq_(word.text, 'text')

    def test_index_property(self):
        word = Word('text', 1000, 100, 10)
        eq_(word.index, 1000)

    def test_count_property(self):
        word = Word('text', 1000, 100, 10)
        eq_(word.count, 100)

    def test_doc_count_property(self):
        word = Word('text', 1000, 100, 10)
        eq_(word.doc_count, 10)

    def test_pickle(self):
        word = Word('text', 1000, 100, 10)
        word2 = pickle.loads(pickle.dumps(word))
        eq_(word2.text, word.text)
        eq_(word2.index, word.index)
        eq_(word2.count, word.count)
        eq_(word2.doc_count, word.doc_count)


class TestEntity(unittest.TestCase):
    def test_title_property(self):
        entity = Entity('Title', 1000, 100, 10)
        eq_(entity.title, 'Title')

    def test_index_property(self):
        entity = Entity('Title', 1000, 100, 10)
        eq_(entity.index, 1000)

    def test_count_property(self):
        entity = Entity('Title', 1000, 100, 10)
        eq_(entity.count, 100)

    def test_doc_count_property(self):
        entity = Entity('Title', 1000, 100, 10)
        eq_(entity.doc_count, 10)

    def test_pickle(self):
        entity = Entity('Title', 1000, 100, 10)
        entity2 = pickle.loads(pickle.dumps(entity))
        eq_(entity2.title, entity.title)
        eq_(entity2.index, entity.index)
        eq_(entity2.count, entity.count)
        eq_(entity2.doc_count, entity.doc_count)


class TestDictionary(unittest.TestCase):
    def setUp(self):
        self.phrase_dic = PhraseDictionary.build(get_dump_db(), min_link_count=0, min_link_prob=0.1,
                                                 lowercase=True, max_phrase_len=3, pool_size=1,
                                                 chunk_size=1, progressbar=False)
        self.dictionary = Dictionary.build(get_dump_db(), phrase_dict=self.phrase_dic,
                                           lowercase=True, min_word_count=2, min_entity_count=1,
                                           min_paragraph_len=5, category=True, pool_size=1,
                                           chunk_size=1, progressbar=False)

    def test_uuid_property(self):
        ok_(isinstance(self.dictionary.uuid, six.text_type))
        eq_(32, len(self.dictionary.uuid))

    def test_language_property(self):
        eq_('en', self.dictionary.language)

    def test_lowercase_property(self):
        eq_(True, self.dictionary.lowercase)

    def test_build_params_property(self):
        build_params = self.dictionary.build_params
        eq_(build_params['dump_db'], get_dump_db().uuid)
        ok_(build_params['dump_file'].endswith('enwiki-pages-articles-sample.xml.bz2'))
        eq_(2, build_params['min_word_count'])
        eq_(1, build_params['min_entity_count'])
        ok_(isinstance(build_params['build_time'], float))
        ok_(build_params['build_time'] > 0)

    def test_min_paragraph_len_property(self):
        eq_(5, self.dictionary.min_paragraph_len)

    def test_entity_offset_property(self):
        eq_(884, self.dictionary.entity_offset)

    def test_word_size_property(self):
        eq_(884, self.dictionary.word_size)

    def test_entity_size_property(self):
        eq_(246, self.dictionary.entity_size)

    def test_len(self):
        eq_(1130, len(self.dictionary))

    def test_iterator(self):
        items = list(self.dictionary)
        eq_(len(self.dictionary), len(items))
        ok_(all(isinstance(item, Item) for item in items))

    def test_words_iterator(self):
        words = list(self.dictionary.words())
        eq_(self.dictionary.word_size, len(words))
        ok_(all(isinstance(word, Word) for word in words))

    def test_entities_iterator(self):
        entities = list(self.dictionary.entities())
        eq_(self.dictionary.entity_size, len(entities))
        ok_(all(isinstance(entity, Entity) for entity in entities))

    def test_get_word(self):
        word = self.dictionary.get_word('the')
        ok_(isinstance(word, Word))
        eq_('the', word.text)
        eq_(190, word.index)
        eq_(427, word.count)
        eq_(2, word.doc_count)

    def test_get_word_not_exist(self):
        eq_(None, self.dictionary.get_word('foobar'))

    def test_get_word_index(self):
        eq_(190, self.dictionary.get_word_index('the'))

    def test_get_word_index_not_exist(self):
        eq_(-1, self.dictionary.get_word_index('foobar'))

    def test_get_entity(self):
        entity = self.dictionary.get_entity('Computer system')
        ok_(isinstance(entity, Entity))
        eq_(1122, entity.index)
        eq_(1, entity.count)
        eq_(1, entity.doc_count)

    def test_get_entity_redirect(self):
        eq_('Computer accessibility', self.dictionary.get_entity('AccessibleComputing').title)
        eq_(None, self.dictionary.get_entity('AccessibleComputing', resolve_redirect=False))

    def test_get_entity_not_exist(self):
        eq_(None, self.dictionary.get_entity('Foo'))

    def test_get_entity_index(self):
        eq_(1122, self.dictionary.get_entity_index('Computer system'))

    def test_get_entity_index_not_exist(self):
        eq_(-1, self.dictionary.get_entity_index('Foo'))

    def test_get_item_by_index(self):
        item = self.dictionary.get_item_by_index(190)
        ok_(isinstance(item, Word))
        eq_('the', item.text)
        eq_(190, item.index)
        eq_(427, item.count)
        eq_(2, item.doc_count)

        item2 = self.dictionary.get_item_by_index(1122)
        ok_(isinstance(item2, Entity))
        eq_('Computer system', item2.title)
        eq_(1122, item2.index)
        eq_(1, item2.count)
        eq_(1, item2.doc_count)

    @raises(KeyError)
    def test_get_item_by_index_not_exist(self):
        self.dictionary.get_item_by_index(100000)

    def test_get_word_by_index(self):
        word = self.dictionary.get_word_by_index(190)
        ok_(isinstance(word, Word))
        eq_('the', word.text)
        eq_(190, word.index)
        eq_(427, word.count)
        eq_(2, word.doc_count)

    @raises(KeyError)
    def test_get_word_by_index_not_exist(self):
        self.dictionary.get_word_by_index(884)

    def test_get_entity_by_index(self):
        entity = self.dictionary.get_entity_by_index(1122)
        ok_(isinstance(entity, Entity))
        eq_('Computer system', entity.title)
        eq_(1122, entity.index)
        eq_(1, entity.count)
        eq_(1, entity.doc_count)

    @raises(KeyError)
    def test_get_entity_by_index_not_exist(self):
        self.dictionary.get_entity_by_index(0)

    def test_get_tokenizer(self):
        tokenizer = self.dictionary.get_tokenizer()
        ok_(isinstance(tokenizer, BaseTokenizer))

    def test_save_load(self):
        def validate(obj):
            s1 = self.dictionary.serialize()
            s2 = obj.serialize()
            for key in s1.keys():
                if isinstance(s1[key], np.ndarray):
                    np.array_equal(s1[key], s2[key])
                else:
                    eq_(s1[key], s2[key])

        validate(Dictionary.load(self.dictionary.serialize()))

        with NamedTemporaryFile() as f:
            self.dictionary.save(f.name)
            validate(Dictionary.load(f.name))
