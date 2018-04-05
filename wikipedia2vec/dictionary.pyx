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
from tqdm import tqdm

from .dump_db cimport DumpDB, Paragraph, WikiLink
from .phrase cimport PhraseDictionary
from .utils.tokenizer import get_tokenizer
from .utils.tokenizer.base_tokenizer cimport BaseTokenizer
from .utils.tokenizer.token cimport Token

logger = logging.getLogger(__name__)

cdef DumpDB _dump_db = None
cdef PhraseDictionary _phrase_dict = None
cdef BaseTokenizer _tokenizer = None


cdef class Item:
    def __init__(self, int32_t index, int32_t count, int32_t doc_count):
        self.index = index
        self.count = count
        self.doc_count = doc_count


cdef class Word(Item):
    def __init__(self, unicode text, int32_t index, int32_t count, int32_t doc_count):
        super(Word, self).__init__(index, count, doc_count)
        self.text = text

    def __repr__(self):
        if six.PY2:
            return b'<Word %s>' % self.text.encode('utf-8')
        else:
            return '<Word %s>' % self.text

    def __reduce__(self):
        return (self.__class__, (self.text, self.index, self.count, self.doc_count))


cdef class Entity(Item):
    def __init__(self, unicode title, int32_t index, int32_t count, int32_t doc_count):
        super(Entity, self).__init__(index, count, doc_count)
        self.title = title

    def __repr__(self):
        if six.PY2:
            return b'<Entity %s>' % self.title.encode('utf-8')
        else:
            return '<Entity %s>' % self.title

    def __reduce__(self):
        return (self.__class__, (self.title, self.index, self.count, self.doc_count))


cdef class Dictionary:
    def __init__(self, word_dict, entity_dict, redirect_dict, PhraseDictionary phrase_dict,
                 np.ndarray word_stats, np.ndarray entity_stats, unicode language, bint lowercase,
                 dict build_params):
        self._word_dict = word_dict
        self._entity_dict = entity_dict
        self._redirect_dict = redirect_dict
        self._phrase_dict = phrase_dict
        self._word_stats = word_stats
        self._entity_stats = entity_stats
        self._language = language
        self._lowercase = lowercase
        self._build_params = build_params

        self._entity_offset = len(self._word_dict)

    @property
    def lowercase(self):
        return self._lowercase

    @property
    def build_params(self):
        return self._build_params

    @property
    def word_offset(self):
        return 0

    @property
    def entity_offset(self):
        return self._entity_offset

    @property
    def word_size(self):
        return len(self._word_dict)

    @property
    def entity_size(self):
        return len(self._entity_dict)

    def __len__(self):
        return len(self._word_dict) + len(self._entity_dict)

    def __iter__(self):
        return chain(self.words(), self.entities())

    def words(self):
        cdef unicode word
        cdef int32_t index

        for (word, index) in six.iteritems(self._word_dict):
            yield Word(word, index, *self._word_stats[index])

    def entities(self):
        cdef unicode title
        cdef int32_t index

        for (title, index) in six.iteritems(self._entity_dict):
            yield Entity(title, index + self._entity_offset, *self._entity_stats[index])

    cpdef get_word(self, unicode word, default=None):
        cdef int32_t index

        index = self.get_word_index(word)
        if index == -1:
            return default
        else:
            return Word(word, index, *self._word_stats[index])

    cpdef int32_t get_word_index(self, unicode word):
        try:
            return self._word_dict[word]
        except KeyError:
            return -1

    cpdef get_entity(self, unicode title, bint resolve_redirect=True, default=None):
        cdef int32_t index, dict_index

        index = self.get_entity_index(title, resolve_redirect=resolve_redirect)
        if index == -1:
            return default
        else:
            dict_index = index - self._entity_offset
            title = self._entity_dict.restore_key(dict_index)
            return Entity(title, index, *self._entity_stats[dict_index])

    cpdef int32_t get_entity_index(self, unicode title, bint resolve_redirect=True):
        cdef int32_t index

        if resolve_redirect:
            try:
                index = self._redirect_dict[title][0][0]
                return index + self._entity_offset
            except KeyError:
                pass
        try:
            index = self._entity_dict[title]
            return index + self._entity_offset
        except KeyError:
            return -1

    cpdef Item get_item_by_index(self, int32_t index):
        if index < self._entity_offset:
            return self.get_word_by_index(index)
        else:
            return self.get_entity_by_index(index)

    cpdef Word get_word_by_index(self, int32_t index):
        cdef unicode word

        word = self._word_dict.restore_key(index)
        return Word(word, index, *self._word_stats[index])

    cpdef Entity get_entity_by_index(self, int32_t index):
        cdef unicode title
        cdef int32_t dict_index

        dict_index = index - self._entity_offset
        title = self._entity_dict.restore_key(dict_index)
        return Entity(title, index, *self._entity_stats[dict_index])

    cpdef BaseTokenizer get_tokenizer(self):
        return get_tokenizer(self._language, phrase_dict=self._phrase_dict)

    @staticmethod
    def build(dump_db, phrase_dict, lowercase, min_word_count, min_entity_count, pool_size,
              chunk_size, progressbar=True):
        global _dump_db, _phrase_dict, _tokenizer

        start_time = time.time()

        if phrase_dict is not None:
            assert phrase_dict.lowercase == lowercase,\
                'Lowercase config must be consistent with PhraseDictionary'

            _phrase_dict = phrase_dict

        _dump_db = dump_db
        _tokenizer = get_tokenizer(dump_db.language, phrase_dict=phrase_dict)

        logger.info('Step 1/3: Processing Wikipedia pages...')

        word_counter = Counter()
        word_doc_counter = Counter()
        entity_counter = Counter()
        entity_doc_counter = Counter()

        with closing(Pool(pool_size)) as pool:
            with tqdm(total=dump_db.page_size(), disable=not progressbar) as bar:
                for (word_cnt, entity_cnt) in pool.imap_unordered(_process_page, dump_db.titles(),
                                                                  chunksize=chunk_size):
                    for (word, count) in word_cnt.items():
                        word_counter[word] += count
                        word_doc_counter[word] += 1

                    for (title, count) in entity_cnt.items():
                        entity_counter[title] += count
                        entity_doc_counter[title] += 1

                    bar.update(1)

        logger.info('Step 2/3: Processing Wikipedia redirects...')

        for (title, dest_title) in dump_db.redirects():
            entity_counter[dest_title] += entity_counter[title]
            del entity_counter[title]

            entity_doc_counter[dest_title] += entity_doc_counter[title]
            del entity_doc_counter[title]

        logger.info('Step 3/3: Building dictionary...')

        word_dict = Trie([w for (w, c) in six.iteritems(word_counter) if c >= min_word_count])
        word_stats = np.zeros((len(word_counter), 2), dtype=np.int32)
        for (word, index) in six.iteritems(word_dict):
            word_stats[index][0] = word_counter[word]
            word_stats[index][1] = word_doc_counter[word]

        del word_counter
        del word_doc_counter

        entity_dict = Trie([e for (e, c) in six.iteritems(entity_counter)
                            if c >= min_entity_count])
        entity_stats = np.zeros((len(entity_counter), 2), dtype=np.int32)
        for (entity, index) in six.iteritems(entity_dict):
            entity_stats[index][0] = entity_counter[entity]
            entity_stats[index][1] = entity_doc_counter[entity]

        del entity_counter
        del entity_doc_counter

        redirect_dict = RecordTrie('<I', [
            (title, (entity_dict[dest_title],))
            for (title, dest_title) in dump_db.redirects() if dest_title in entity_dict
        ])

        if phrase_dict is not None:
            phrase_trie = Trie([phrase for phrase in phrase_dict if phrase in word_dict])
            phrase_dict.phrase_trie = phrase_trie

        build_params = dict(
            dump_file=dump_db.dump_file,
            min_word_count=min_word_count,
            min_entity_count=min_entity_count,
            build_time=time.time() - start_time,
        )

        return Dictionary(word_dict, entity_dict, redirect_dict, phrase_dict, word_stats,
                          entity_stats, dump_db.language, lowercase, build_params)

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    def serialize(self):
        obj = dict(
            word_dict=self._word_dict.tobytes(),
            entity_dict=self._entity_dict.tobytes(),
            redirect_dict=self._redirect_dict.tobytes(),
            word_stats=np.asarray(self._word_stats, dtype=np.int32),
            entity_stats=np.asarray(self._entity_stats, dtype=np.int32),
            meta=dict(language=self._language,
                      lowercase=self._lowercase,
                      build_params=self._build_params)
        )
        if self._phrase_dict is not None:
            obj['phrase_dict'] = self._phrase_dict.serialize()

        return obj

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
        if 'phrase_dict' in target:
            phrase_dict = PhraseDictionary.load(target['phrase_dict'])
        else:
            phrase_dict = None

        return Dictionary(word_dict, entity_dict, redirect_dict, phrase_dict, target['word_stats'],
                          target['entity_stats'], **target['meta'])


def _process_page(unicode title):
    cdef Paragraph paragraph
    cdef Token token
    cdef WikiLink link

    word_counter = Counter()
    entity_counter = Counter()

    for paragraph in _dump_db.get_paragraphs(title):
        word_counter.update(token.text for token in _tokenizer.tokenize(paragraph.text))
        entity_counter.update(link.title for link in paragraph.wiki_links)

    return (word_counter, entity_counter)
