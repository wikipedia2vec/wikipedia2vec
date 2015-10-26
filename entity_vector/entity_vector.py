# -*- coding: utf-8 -*-

import cPickle as pickle
import logging
import multiprocessing
import numpy as np
import os
from annoy import AnnoyIndex
from ctypes import c_float, c_uint, c_uint64
from functools import partial
from itertools import imap
from multiprocessing.pool import Pool

from . import REAL
from dictionary import Item, Entity, Word
from entity_vector_worker import init_worker, train_page

logger = logging.getLogger(__name__)


def cosine(vec1, vec2):
    return np.dot(vec1, vec2) / np.linalg.norm(vec1) / np.linalg.norm(vec2)


class EntityVector(object):
    def __init__(
        self, dictionary, size=300, word_alpha=0.025, word_min_alpha=0.0001,
        entity_alpha=0.025, entity_min_alpha=0.0001, word_window=10,
        word_negative=30, entity_negative=30, iteration=10
    ):
        self.dictionary = dictionary
        self.syn0 = None
        self.syn1 = None

        self._size = size
        self._word_initial_alpha = word_alpha
        self._word_min_alpha = word_min_alpha
        self._entity_initial_alpha = entity_alpha
        self._entity_min_alpha = entity_min_alpha
        self._word_window = word_window
        self._word_negative = word_negative
        self._entity_negative = entity_negative
        self._iteration = iteration

        self._word_neg_table = None
        self._entity_neg_table = None
        self._annoy_index = None

        self._word_alpha = None
        self._entity_alpha = None
        self._total_words = None
        self._word_counter = None

    def __reduce__(self):
        return (
            self.__class__, (
                self.dictionary,
                self._size,
                self._word_initial_alpha,
                self._word_min_alpha,
                self._entity_initial_alpha,
                self._entity_min_alpha,
                self._word_window,
                self._word_negative,
                self._entity_negative,
                self._iteration,
            )
        )

    def get_word(self, word):
        return self.dictionary.get_word(word)

    def get_entity(self, title):
        return self.dictionary.get_entity(title)

    def get_word_vector(self, word):
        word = self.dictionary.get_word(word)
        return self.syn0[word.index]

    def get_entity_vector(self, title):
        entity = self.dictionary.get_entity(title)
        return self.syn0[entity.index]

    def get_vector(self, item):
        return self.syn0[item.index]

    def get_similarity(self, item1, item2):
        return cosine(self.syn0[item1.index], self.syn0[item2.index])

    def __getitem__(self, key):
        if isinstance(key, Item):
            return self.syn0[key.index]

        elif isinstance(key, unicode):
            try:
                word = self.dictionary.get_word(key)
                return self.syn0[word.index]
            except KeyError:
                pass

            try:
                entity = self.dictionary.get_entity(key)
                return self.syn0[entity.index]
            except KeyError:
                pass

        elif isinstance(key, int):
            return self.syn0[key]

        raise KeyError()

    def most_similar(self, item, count=100, search_k=-1, recalc_sim=True):
        return self.most_similar_by_vector(self.get_vector(item), count,
                                           search_k, recalc_sim)

    def most_similar_by_vector(self, vec, count=100, search_k=-1,
                               recalc_sim=True):
        if not self._annoy_index:
            raise RuntimeError('Failed to find the vector index')

        ret = [
            (self.dictionary[i], dist)
            for (i, dist) in zip(*self._annoy_index.get_nns_by_vector(
                vec, count, search_k, True
            ))
        ]
        if recalc_sim:
            ret = sorted(
                [(o[0], cosine(vec, self.get_vector(o[0]))) for o in ret],
                key=lambda k: k[1], reverse=True
            )

        return ret

    def __len__(self):
        return len(self.dictionary)

    def __iter__(self):
        for item in self.dictionary.iteritems():
            yield (item, self.syn0[item.index])

    def save(self, out_file):
        if out_file.endswith('.pickle'):
            out_file = out_file[:-7]

        with open(out_file + '.pickle', 'w') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

        if self._word_neg_table:
            with open(out_file + '_word_neg.pickle', 'w') as f:
                pickle.dump(list(self._word_neg_table), f,
                            protocol=pickle.HIGHEST_PROTOCOL)
        if self._entity_neg_table:
            with open(out_file + '_entity_neg.pickle', 'w') as f:
                pickle.dump(list(self._entity_neg_table), f,
                            protocol=pickle.HIGHEST_PROTOCOL)

        if self._annoy_index:
            self._annoy_index.save(out_file + '.ann')

        if self.syn0 is not None:
            np.save(out_file + '_syn0.npy', self.syn0)
        if self.syn1 is not None:
            np.save(out_file + '_syn1.npy', self.syn1)

    @staticmethod
    def load(in_file, numpy_mmap_mode='r+'):
        if in_file.endswith('.pickle'):
            in_file = in_file[:-7]

        with open(in_file + '.pickle') as f:
            ret = pickle.load(f)

        if os.path.isfile(in_file + '_word_neg.pickle'):
            with open(in_file + '_word_neg.pickle') as f:
                ret._word_neg_table = multiprocessing.RawArray(
                    c_uint, pickle.load(f)
                )
        if os.path.isfile(in_file + '_entity_neg.pickle'):
            with open(in_file + '_entity_neg.pickle') as f:
                ret._entity_neg_table = multiprocessing.RawArray(
                    c_uint, pickle.load(f)
                )

        if os.path.isfile(in_file + '.ann'):
            ret._annoy_index = AnnoyIndex(ret._size, 'angular')
            ret._annoy_index.load(in_file + '.ann')

        if os.path.isfile(in_file + '_syn0.npy'):
            ret.syn0 = np.load(in_file + '_syn0.npy', mmap_mode=numpy_mmap_mode)
        if os.path.isfile(in_file + '_syn1.npy'):
            ret.syn1 = np.load(in_file + '_syn1.npy', mmap_mode=numpy_mmap_mode)

        return ret

    def train(self, dump_reader, parallel=True,
              pool_size=multiprocessing.cpu_count(), chunk_size=100):
        self._word_counter = multiprocessing.Value(c_uint64, 0)
        self._word_alpha = multiprocessing.RawValue(
            c_float, self._word_initial_alpha
        )
        self._entity_alpha = multiprocessing.RawValue(
            c_float, self._entity_initial_alpha
        )

        logger.info('Initializing weights...')
        syn0_shared = multiprocessing.RawArray(
            c_float, len(self.dictionary) * self._size
        )
        syn0 = np.frombuffer(syn0_shared, dtype=REAL)
        syn0 = syn0.reshape(len(self.dictionary), self._size)
        for w in self.dictionary:
            if isinstance(w, Word):
                np.random.seed(np.uint32(hash(w.text)))
            elif isinstance(w, Entity):
                np.random.seed(np.uint32(hash(w.title)))
            else:
                RuntimeError('Unknown type')

            syn0[w.index] = (np.random.rand(self._size) - 0.5) / self._size

        syn1_shared = multiprocessing.RawArray(
            c_float, len(self.dictionary) * self._size
        )
        syn1 = np.frombuffer(syn1_shared, dtype=REAL)
        syn1 = syn1.reshape(len(self.dictionary), self._size)
        syn1.fill(0)

        self._total_words = int(sum(
            w.count for w in self.dictionary.words()
        ))
        self._total_words *= self._iteration
        logger.info('Total number of words: %d', self._total_words)

        word_neg_table = self._build_word_neg_table()
        entity_neg_table = self._build_entity_neg_table()

        logger.info('Starting to train a model...')

        def iter_dump_reader():
            for n in range(self._iteration):
                logger.info('Iteration: %d', n)
                for page in dump_reader:
                    yield page

        init_args = (
            self, syn0_shared, syn1_shared, word_neg_table, entity_neg_table
        )

        if parallel:
            pool = Pool(pool_size, initializer=init_worker, initargs=init_args)
            imap_func = partial(pool.imap_unordered, chunksize=chunk_size)
        else:
            init_worker(*init_args)
            imap_func = imap

        for (n, _) in enumerate(imap_func(train_page, iter_dump_reader())):
            if n % 10000 == 0:
                prog = float(self._word_counter.value) / self._total_words
                logger.info(
                    'Proccessing page #%d progress: %.1f%% '
                    'word alpha: %.3f entity alpha: %.3f',
                    n, prog * 100, self._word_alpha.value,
                    self._entity_alpha.value
                )

        if parallel:
            pool.close()

        self.syn0 = syn0
        self.syn1 = syn1
        self._word_neg_table = word_neg_table
        self._entity_neg_table = entity_neg_table

    def init_sims(self):
        logger.info('Precomputing L2-norms of word vectors')

        for i in xrange(self.syn0.shape[0]):
            self.syn0[i, :] /= np.sqrt((self.syn0[i, :] ** 2).sum(-1))

    def build_vector_index(self, n_trees=10):
        logger.info('Creating Annoy search index...')

        index = AnnoyIndex(self._size, 'angular')
        for (i, v) in enumerate(self.syn0):
            index.add_item(i, v)

        index.build(n_trees)
        self._annoy_index = index

    def _build_word_neg_table(self, table_size=100000000, power=0.75):
        logger.info('Creating word index table for negative sampling...')

        return EntityVector._build_unigram_neg_table(
            list(self.dictionary.words()), table_size, power
        )

    def _build_entity_neg_table(self):
        logger.info('Creating entity index table for negative sampling...')

        # uniform distribution over all entities
        return multiprocessing.RawArray(
            c_uint, [e.index for e in self.dictionary.entities()]
        )

    @staticmethod
    def _build_unigram_neg_table(items, table_size, power):
        neg_table = multiprocessing.RawArray(c_uint, table_size)

        items_pow = float(sum([
            item.count ** power for item in items
        ]))

        index = 0
        cur = items[index].count ** power / items_pow

        for table_index in xrange(table_size):
            neg_table[table_index] = items[index].index
            if float(table_index) / table_size > cur:
                if index < len(items) - 1:
                    index += 1
                cur += items[index].count ** power / items_pow

        return neg_table
