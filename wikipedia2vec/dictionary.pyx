# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import joblib
import logging
import multiprocessing
import time
import six
import six.moves.cPickle as pickle
import numpy as np
from collections import Counter
from contextlib import closing
from functools import partial
from itertools import chain
from marisa_trie import Trie, RecordTrie
from multiprocessing.pool import Pool

from .extractor import Extractor
from .utils.wiki_page cimport WikiPage

logger = logging.getLogger(__name__)

_extractor = None


cdef class Item:
    def __init__(self, int index, int count, int doc_count):
        self._index = index
        self._count = count
        self._doc_count = doc_count

    property index:
        def __get__(self):
            return self._index

    property count:
        def __get__(self):
            return self._count

    property doc_count:
        def __get__(self):
            return self._doc_count


cdef class Word(Item):
    def __init__(self, unicode text, int index, int count, int doc_count):
        super(Word, self).__init__(index, count, doc_count)
        self._text = text

    property text:
        def __get__(self):
            return self._text

    def __repr__(self):
        if six.PY2:
            return b'<Word %s>' % self._text.encode('utf-8')
        else:
            return '<Word %s>' % self._text

    def __reduce__(self):
        return (self.__class__, (self._text, self._index, self._count, self._doc_count))


cdef class Entity(Item):
    def __init__(self, unicode title, int index, int count, int doc_count):
        super(Entity, self).__init__(index, count, doc_count)
        self._title = title

    property title:
        def __get__(self):
            return self._title

    def __repr__(self):
        if six.PY2:
            return b'<Entity %s>' % self._title.encode('utf-8')
        else:
            return '<Entity %s>' % self._title

    def __reduce__(self):
        return (self.__class__, (self._title, self._index, self._count, self._doc_count))


cdef class PrefixSearchable:
    cpdef list prefix_search(self, unicode text, int start=0):
        raise NotImplementedError()


cdef class Dictionary(PrefixSearchable):
    def __init__(self, word_dict, entity_dict, redirect_dict, np.ndarray word_stats,
                 np.ndarray entity_stats, bint lowercase, dict build_params):
        self._word_dict = word_dict
        self._entity_dict = entity_dict
        self._redirect_dict = redirect_dict
        self._word_stats = word_stats
        self._entity_stats = entity_stats
        self._lowercase = lowercase
        self._build_params = build_params

        self._entity_offset = len(word_dict)

    property lowercase:
        def __get__(self):
            return self._lowercase

    property build_params:
        def __get__(self):
            return self._build_params

    property entity_offset:
        def __get__(self):
            return self._entity_offset

    def __reduce__(self):
        return (self.__class__, (
            self._word_dict, self._entity_dict, self._redirect_dict, self._word_stats,
            self._entity_stats, self._lowercase, self._build_params
        ))

    cpdef get_word(self, unicode word, default=None):
        cdef int index
        index = self.get_word_index(word)
        if index == -1:
            return default
        else:
            return Word(word, index, *self._word_stats[index])

    cpdef int get_word_index(self, unicode word):
        cdef int index
        try:
            return self._word_dict[word]
        except KeyError:
            return -1

    cpdef get_entity(self, unicode title, bint resolve_redirect=True,
                     default=None):
        cdef int index, dict_index

        index = self.get_entity_index(title, resolve_redirect=resolve_redirect)
        if index == -1:
            return default
        else:
            dict_index = index - self._entity_offset
            title = self._entity_dict.restore_key(dict_index)
            return Entity(title, index, *self._entity_stats[dict_index])

    cpdef int get_entity_index(self, unicode title, bint resolve_redirect=True):
        cdef int index

        if resolve_redirect:
            try:
                return self._redirect_dict[title][0][0] + self._entity_offset
            except KeyError:
                pass
        try:
            return self._entity_dict[title] + self._entity_offset
        except KeyError:
            return -1

    cpdef Item get_item_by_index(self, int index):
        if index < self._entity_offset:
            return self.get_word_by_index(index)
        else:
            return self.get_entity_by_index(index)

    cpdef Word get_word_by_index(self, int index):
        cdef unicode word
        word = self._word_dict.restore_key(index)
        return Word(word, index, *self._word_stats[index])

    cpdef Entity get_entity_by_index(self, int index):
        cdef unicode title
        cdef int dict_index

        dict_index = index - self._entity_offset
        title = self._entity_dict.restore_key(dict_index)
        return Entity(title, index, *self._entity_stats[dict_index])

    cpdef list prefix_search(self, unicode text, int start=0):
        return sorted(self._word_dict.prefixes(text[start:]), key=len, reverse=True)

    def __len__(self):
        return len(self._word_dict) + len(self._entity_dict)

    def __iter__(self):
        return chain(self.words(), self.entities())

    def words(self):
        cdef unicode word
        cdef int index

        for (word, index) in six.iteritems(self._word_dict):
            yield Word(word, index, *self._word_stats[index])

    def entities(self):
        cdef unicode title
        cdef int index

        for (title, index) in six.iteritems(self._entity_dict):
            yield Entity(title, index + self._entity_offset, *self._entity_stats[index])

    @staticmethod
    def build(dump_reader, phrase_dict, lowercase, min_word_count, min_entity_count, pool_size,
              chunk_size):
        logger.info('Starting to build a dictionary')

        start_time = time.time()

        if phrase_dict:
            assert phrase_dict.lowercase == lowercase,\
                'Lowercase config must be consistent with PhraseDictionary'

        word_counter = Counter()
        word_doc_counter = Counter()
        entity_counter = Counter()
        entity_doc_counter = Counter()
        entity_redirects = {}

        logger.info('Step 1/3: Processing Wikipedia pages...')

        global _extractor
        _extractor = Extractor(dump_reader.language, lowercase=lowercase, dictionary=phrase_dict)

        with closing(Pool(pool_size)) as pool:
            for (page, paragraphs) in pool.imap_unordered(_extract_paragraphs, dump_reader,
                                                          chunksize=chunk_size):
                if page.is_redirect:
                    entity_redirects[page.title] = page.redirect

                else:
                    words = []
                    entities = []
                    for paragraph in paragraphs:
                        words += paragraph.words
                        entities += [l.title for l in paragraph.wiki_links]

                    word_counter.update(words)
                    entity_counter.update(entities)
                    word_doc_counter.update(list(set(words)))
                    entity_doc_counter.update(list(set(entities)))

        _extractor = None

        logger.info('Step 2/3: Processing Wikipedia redirects...')

        for (title, dest_title) in six.iteritems(entity_redirects):
            entity_counter[dest_title] += entity_counter[title]
            del entity_counter[title]

            entity_doc_counter[dest_title] += entity_doc_counter[title]
            del entity_doc_counter[title]

        logger.info('Step 3/3: Building dictionary...')

        word_dict = Trie([w for (w, c) in six.iteritems(word_counter) if c >= min_word_count])
        word_stats = np.zeros((len(word_counter), 2), dtype=np.int)
        for (word, index) in six.iteritems(word_dict):
            word_stats[index][0] = word_counter[word]
            word_stats[index][1] = word_doc_counter[word]

        del word_counter
        del word_doc_counter

        entity_dict = Trie([e for (e, c) in six.iteritems(entity_counter)
                            if c >= min_entity_count])
        entity_stats = np.zeros((len(entity_counter), 2), dtype=np.int)
        for (entity, index) in six.iteritems(entity_dict):
            entity_stats[index][0] = entity_counter[entity]
            entity_stats[index][1] = entity_doc_counter[entity]

        del entity_counter
        del entity_doc_counter

        redirect_items = []
        for (title, dest_title) in six.iteritems(entity_redirects):
            if dest_title in entity_dict:
                redirect_items.append((title, (entity_dict[dest_title],)))

        redirect_dict = RecordTrie('<I', redirect_items)

        if phrase_dict is None:
            phrase_dict_params = None
        else:
            phrase_dict_params = dict(build_params=phrase_dict.build_params)

        build_params = dict(
            dump_file=dump_reader.dump_file,
            min_word_count=min_word_count,
            min_entity_count=min_entity_count,
            phrase_dict=phrase_dict_params,
            build_time=time.time() - start_time,
        )

        return Dictionary(word_dict, entity_dict, redirect_dict, word_stats, entity_stats,
                          lowercase, build_params)

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    def serialize(self):
        return dict(
            word_dict=self._word_dict.tobytes(),
            entity_dict=self._entity_dict.tobytes(),
            redirect_dict=self._redirect_dict.tobytes(),
            word_stats=self._word_stats,
            entity_stats=self._entity_stats,
            meta=dict(lowercase=self._lowercase, build_params=self._build_params)
        )

    @staticmethod
    def load(target, mmap=True):
        word_dict = Trie()
        entity_dict = Trie()
        redirect_dict = RecordTrie('<I')

        if not isinstance(target, dict):
            if mmap:
                target = joblib.load(target, mmap_mode='r')
            else:
                target = joblib.load(target)

        word_dict.frombytes(target['word_dict'])
        entity_dict.frombytes(target['entity_dict'])
        redirect_dict.frombytes(target['redirect_dict'])

        return Dictionary(word_dict, entity_dict, redirect_dict, target['word_stats'],
                          target['entity_stats'], **target['meta'])


def _extract_paragraphs(WikiPage page):
    try:
        return (page, _extractor.extract_paragraphs(page))
    except:
        logging.exception('Unknown exception')
        raise
