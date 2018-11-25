# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import joblib
import logging
import multiprocessing
import numpy as np
import pkg_resources
import time
import six
import six.moves.cPickle as pickle
from collections import Counter
from contextlib import closing
from functools import partial
from itertools import chain
from marisa_trie import Trie, RecordTrie
from multiprocessing.pool import Pool
from tqdm import tqdm
from uuid import uuid1

from .dump_db cimport DumpDB, Paragraph, WikiLink
from .utils.tokenizer.token cimport Token

logger = logging.getLogger(__name__)


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
    def __init__(self, word_dict, entity_dict, redirect_dict, np.ndarray word_stats,
                 np.ndarray entity_stats, unicode language, bint lowercase, dict build_params,
                 int32_t min_paragraph_len=0, unicode uuid=''):
        self._word_dict = word_dict
        self._entity_dict = entity_dict
        self._redirect_dict = redirect_dict
        # Limit word_stats size in order to handle existing pretrained embeddings which is larger than it should be
        self._word_stats = word_stats[:len(self._word_dict)]
        self._entity_stats = entity_stats[:len(self._entity_dict)]
        self.min_paragraph_len = min_paragraph_len
        self.uuid = uuid
        self.language = language
        self.lowercase = lowercase
        self.build_params = build_params

        self._entity_offset = len(self._word_dict)

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

    @staticmethod
    def build(dump_db, tokenizer, lowercase, min_word_count, min_entity_count, min_paragraph_len,
              category, disambi, pool_size, chunk_size, progressbar=True):
        start_time = time.time()

        logger.info('Step 1/2: Processing Wikipedia pages...')

        word_counter = Counter()
        word_doc_counter = Counter()
        entity_counter = Counter()
        entity_doc_counter = Counter()

        with closing(Pool(pool_size, initializer=init_worker, initargs=(dump_db, tokenizer))) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_process_page, lowercase=lowercase, min_paragraph_len=min_paragraph_len)
                for (word_cnt, entity_cnt) in pool.imap_unordered(f, dump_db.titles(),
                                                                  chunksize=chunk_size):
                    for (word, count) in word_cnt.items():
                        word_counter[word] += count
                        word_doc_counter[word] += 1

                    for (title, count) in entity_cnt.items():
                        if '#' in title:
                            continue
                        if not category and title.startswith('Category:'):
                            continue
                        entity_counter[title] += count
                        entity_doc_counter[title] += 1

                    bar.update(1)

        logger.info('Step 2/2: Processing Wikipedia redirects...')

        for (title, dest_title) in dump_db.redirects():
            entity_counter[dest_title] += entity_counter[title]
            del entity_counter[title]

            entity_doc_counter[dest_title] += entity_doc_counter[title]
            del entity_doc_counter[title]

        word_dict = Trie([w for (w, c) in six.iteritems(word_counter) if c >= min_word_count])
        word_stats = np.zeros((len(word_dict), 2), dtype=np.int32)
        for (word, index) in six.iteritems(word_dict):
            word_stats[index][0] = word_counter[word]
            word_stats[index][1] = word_doc_counter[word]

        del word_counter
        del word_doc_counter

        entities = []
        for (entity, count) in six.iteritems(entity_counter):
            if count < min_entity_count:
                continue

            if not disambi and dump_db.is_disambiguation(entity):
                continue

            entities.append(entity)

        entity_dict = Trie(entities)
        entity_stats = np.zeros((len(entity_dict), 2), dtype=np.int32)
        for (entity, index) in six.iteritems(entity_dict):
            entity_stats[index][0] = entity_counter[entity]
            entity_stats[index][1] = entity_doc_counter[entity]

        del entity_counter
        del entity_doc_counter

        redirect_dict = RecordTrie('<I', [
            (title, (entity_dict[dest_title],))
            for (title, dest_title) in dump_db.redirects() if dest_title in entity_dict
        ])

        build_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            min_word_count=min_word_count,
            min_entity_count=min_entity_count,
            category=category,
            build_time=time.time() - start_time,
            version=pkg_resources.get_distribution('wikipedia2vec').version
        )

        uuid = six.text_type(uuid1().hex)

        logger.info('%d words and %d entities are indexed in the dictionary', len(word_dict),
                    len(entity_dict))

        return Dictionary(word_dict, entity_dict, redirect_dict, word_stats, entity_stats,
                          dump_db.language, lowercase, build_params, min_paragraph_len, uuid)

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    def serialize(self, shared_array=False):
        cdef int32_t [:] word_stats_src, entity_stats_src, word_stats_dst, entity_stats_dst

        if shared_array:
            word_stats_src = np.asarray(self._word_stats, dtype=np.int32).flatten()
            entity_stats_src = np.asarray(self._entity_stats, dtype=np.int32).flatten()

            word_stats = multiprocessing.RawArray('i', word_stats_src.size)
            entity_stats = multiprocessing.RawArray('i', entity_stats_src.size)
            word_stats_dst = word_stats
            entity_stats_dst = entity_stats

            word_stats_dst[:] = word_stats_src
            entity_stats_dst[:] = entity_stats_src
        else:
            word_stats = np.asarray(self._word_stats, dtype=np.int32)
            entity_stats = np.asarray(self._entity_stats, dtype=np.int32)

        return dict(
            word_dict=self._word_dict.tobytes(),
            entity_dict=self._entity_dict.tobytes(),
            redirect_dict=self._redirect_dict.tobytes(),
            word_stats=word_stats,
            entity_stats=entity_stats,
            meta=dict(uuid=self.uuid,
                      language=self.language,
                      lowercase=self.lowercase,
                      min_paragraph_len=self.min_paragraph_len,
                      build_params=self.build_params)
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

        word_stats = target['word_stats']
        entity_stats = target['entity_stats']
        if not isinstance(word_stats, np.ndarray):
            word_stats = np.frombuffer(word_stats, dtype=np.int32).reshape(-1, 2)
            entity_stats = np.frombuffer(entity_stats, dtype=np.int32).reshape(-1, 2)

        return Dictionary(word_dict, entity_dict, redirect_dict, word_stats, entity_stats,
                          **target['meta'])


cdef DumpDB _dump_db = None
cdef _tokenizer = None


def init_worker(dump_db, tokenizer):
    global _dump_db, _tokenizer

    _dump_db = dump_db
    _tokenizer = tokenizer


def _process_page(unicode title, bint lowercase, int32_t min_paragraph_len):
    cdef list tokens
    cdef Paragraph paragraph
    cdef Token token
    cdef WikiLink link

    word_counter = Counter()
    entity_counter = Counter()

    for paragraph in _dump_db.get_paragraphs(title):
        entity_counter.update(link.title for link in paragraph.wiki_links)

        tokens = _tokenizer.tokenize(paragraph.text)
        if len(tokens) >= min_paragraph_len:
            if lowercase:
                word_counter.update(token.text.lower() for token in tokens)
            else:
                word_counter.update(token.text for token in tokens)

    return (word_counter, entity_counter)
