# -*- coding: utf-8 -*-

import dawg
import logging
import math
import multiprocessing
import cPickle as pickle
import numpy as np
import uuid
from collections import Counter
from contextlib import closing
from cStringIO import StringIO
from functools import partial
from itertools import imap
from multiprocessing.pool import Pool
from tempfile import NamedTemporaryFile

cimport wiki_page
from wiki_page cimport WikiPage

logger = logging.getLogger(__name__)

cdef str WORD_TYPE = 'w'
cdef str ENTITY_TYPE = 'e'


cdef class Item:
    def __init__(self, int index, int count, int doc_count):
        self.index = index
        self.count = count
        self.doc_count = doc_count


cdef class Word(Item):
    def __init__(self, unicode text, int index, int count, int doc_count):
        super(Word, self).__init__(index, count, doc_count)
        self.text = text

    def __repr__(self):
        return '<Word %s>' % (self.text.encode('utf-8'))

    def __reduce__(self):
        return (
            self.__class__,
            (self.text, self.index, self.count, self.doc_count)
        )


cdef class Entity(Item):
    def __init__(self, unicode title, int index, int count, int doc_count):
        super(Entity, self).__init__(index, count, doc_count)
        self.title = title

    def __repr__(self):
        return '<Entity %s>' % (self.title.encode('utf-8'))

    def __reduce__(self):
        return (
            self.__class__,
            (self.title, self.index, self.count, self.doc_count)
        )


cdef class Dictionary:
    @property
    def id(self):
        return self._id

    @property
    def total_docs(self):
        return self._total_docs

    @property
    def ordered_keys(self):
        if self._ordered_keys is None:
            self._ordered_keys = np.empty(len(self), dtype=np.object_)

            for (key, (_, index, _, _)) in self._dict.iteritems():
                self._ordered_keys[index] = key

        return self._ordered_keys

    cpdef Word get_word(self, unicode word):
        return self._create_item_from_key(word)

    cpdef Entity get_entity(self, unicode title, bint resolve_redirect=True):
        cdef int index
        cdef unicode key

        if resolve_redirect and title in self._redirect_dict:
            index = self._redirect_dict[title][0][0]
            return self[index]

        else:
            key = self._create_entity_key(title)
            return self._create_item_from_key(key)

    def words(self):
        cdef unicode key
        cdef str item_type

        for (key, (item_type, _, _, _)) in self._dict.iteritems():
            if item_type == WORD_TYPE:
                yield self._create_item_from_key(key)

    def entities(self):
        cdef unicode key
        cdef str item_type

        for (key, (item_type, _, _, _)) in self._dict.iteritems():
            if item_type == ENTITY_TYPE:
                yield self._create_item_from_key(key)

    cpdef list get_bow_vector(self, words, bint tfidf=True,
                              bint normalize=True):
        cdef int c
        cdef float norm, weight
        cdef unicode s
        cdef list word_counts, weights
        cdef Word w

        word_counts = [
            (self[s.lower()], c) for (s, c) in Counter(words).items()
            if s.lower() in self
        ]
        if tfidf:
            weights = [
                math.log(c + 1) * math.log(float(self._total_docs) / w.doc_count)
                for (w, c) in word_counts
            ]
        else:
            weights = [c for (_, c) in word_counts]

        if normalize:
            norm = math.sqrt(sum([weight * weight for weight in weights]))
            weights = [weight / norm for weight in weights]

        return [(w, weight) for ((w, _), weight) in zip(word_counts, weights)]

    @staticmethod
    def build(dump_reader, min_word_count, min_entity_count, parallel,
              pool_size=multiprocessing.cpu_count(), chunk_size=100):
        logger.info('Starting to build a dictionary')

        total_docs = 0
        word_counter = Counter()
        word_doc_counter = Counter()
        entity_counter = Counter()
        entity_doc_counter = Counter()
        entity_redirects = {}

        cdef list paragraph, paragraphs
        cdef WikiPage page

        logger.info('Step 1/3: Processing Wikipedia pages...')
        if parallel:
            pool = Pool(pool_size)
            imap_func = partial(pool.imap_unordered, chunksize=chunk_size)
        else:
            imap_func = imap

        for (page, paragraphs) in imap_func(_extract_paragraphs, dump_reader):
            if page.is_redirect:
                entity_redirects[page.title] = page.redirect

            else:
                words = []
                entities = []
                for paragraph in paragraphs:
                    for item in paragraph:
                        if isinstance(item, wiki_page.Word):
                            words.append(item.text.lower())

                        elif isinstance(item, wiki_page.WikiLink):
                            entities.append(item.title)
                            words += [w.text.lower() for w in item.words]

                word_counter.update(words)
                entity_counter.update(entities)
                word_doc_counter.update(list(set(words)))
                entity_doc_counter.update(list(set(entities)))

                total_docs += 1

        if parallel:
            pool.close()

        ret = Dictionary()
        ret._id = uuid.uuid1().hex
        ret._total_docs = total_docs

        logger.info('Step 2/3: Handling Wikipedia redirects...')
        # TODO: multiple redirects should be handled here
        for (title, dest_title) in entity_redirects.iteritems():
            entity_counter[dest_title] += entity_counter[title]
            del entity_counter[title]

            entity_doc_counter[dest_title] += entity_doc_counter[title]
            del entity_doc_counter[title]

        logger.info('Step 3/3: Building DAWG dictionaries...')
        items = []
        index = 0

        for (word, count) in word_counter.iteritems():
            if count >= min_word_count:
                doc_count = word_doc_counter[word]
                items.append((word, (WORD_TYPE, index, count, doc_count)))
                index += 1

        for (title, count) in entity_counter.iteritems():
            if count >= min_entity_count:
                key = ret._create_entity_key(title)
                doc_count = entity_doc_counter[title]
                items.append((key, (ENTITY_TYPE, index, count, doc_count)))
                index += 1

        ret._dict = dawg.RecordDAWG('<cIII', items)
        ret._size = index

        del word_counter
        del word_doc_counter
        del entity_counter
        del entity_doc_counter
        del items

        redirect_items = []
        for (title, dest_title) in entity_redirects.iteritems():
            key = ret._create_entity_key(dest_title)
            if key in ret._dict:
                redirect_items.append((title, (ret._dict[key][0][1],)))

        ret._redirect_dict = dawg.RecordDAWG('<I', redirect_items)

        return ret

    @staticmethod
    def load(in_file):
        return pickle.load(in_file)

    def save(self, out_file):
        pickle.dump(self, out_file, pickle.HIGHEST_PROTOCOL)

    def __len__(self):
        return self._size

    def __contains__(self, k):
        return bool(k in self._dict)

    def __iter__(self):
        cdef unicode key

        for key in self._dict.iterkeys():
            yield self._create_item_from_key(key)

    def __getitem__(self, key):
        if isinstance(key, int):
            return self._create_item_from_key(self.ordered_keys[key])

        elif isinstance(key, unicode):
            try:
                return self.get_word(key)
            except KeyError:
                pass

            try:
                return self.get_entity(key)
            except KeyError:
                pass

        raise KeyError(key)

    def __reduce__(self):
        return (
            self.__class__, tuple(), dict(
                dict=self._dict.tobytes(),
                redirect_dict=self._redirect_dict.tobytes(),
                size=self._size,
                id=self._id,
                total_docs=self._total_docs
            )
        )

    def __setstate__(self, state):
        # We need to use a file here since DAWG has an issue with pickle:
        # http://dawg.readthedocs.org/en/latest/#persistence
        with NamedTemporaryFile('wb') as f:
            f.write(state['dict'])
            f.flush()
            self._dict = dawg.RecordDAWG('<cIII')
            self._dict.load(f.name)

        with NamedTemporaryFile('wb') as f:
            f.write(state['redirect_dict'])
            f.flush()
            self._redirect_dict = dawg.RecordDAWG('<I')
            self._redirect_dict.load(f.name)

        self._size = state['size']
        self._id = state['id']
        self._total_docs = state['total_docs']

    cdef inline Item _create_item_from_key(self, unicode key):
        cdef str item_type
        cdef int index, count

        (item_type, index, count, doc_count) = self._dict[key][0]
        if item_type == WORD_TYPE:
            return Word(key, index, count, doc_count)

        if item_type == ENTITY_TYPE:
            return Entity(self._get_title_from_key(key), index, count,
                          doc_count)

    cdef inline unicode _create_entity_key(self, unicode title):
        return u'ENTITY/' + title

    cdef inline unicode _get_title_from_key(self, unicode key):
        return key[7:]


def _extract_paragraphs(page):
    return (page, list(page.extract_paragraphs()))
