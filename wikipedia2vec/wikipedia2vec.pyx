# -*- coding: utf-8 -*-

import joblib
import logging
import multiprocessing
import os
import time
import six
import six.moves.cPickle as pickle
import numpy as np
cimport numpy as np
cimport cython
from ctypes import c_float, c_uint32, c_uint64
from contextlib import closing
from libc.math cimport exp
from libc.string cimport memset
from multiprocessing.pool import Pool
from scipy.linalg cimport cython_blas as blas

from .dictionary cimport Dictionary, Item, Word, Entity
from .extractor cimport Extractor, Paragraph, WikiLink
from .link_graph cimport LinkGraph
from .utils.wiki_page cimport WikiPage

logger = logging.getLogger(__name__)

cdef int ONE = 1
cdef np.float32_t ONEF = <np.float32_t>1.0
cdef int MAX_EXP = 6

cdef Dictionary dictionary
cdef LinkGraph link_graph
cdef np.ndarray syn0
cdef np.ndarray syn1
cdef np.ndarray work
cdef np.ndarray word_neg_table
cdef np.ndarray entity_neg_table
cdef np.ndarray sample_ints
cdef np.ndarray link_indices
cdef Extractor extractor
cdef unicode language
cdef object alpha
cdef object word_counter
cdef object link_cursor
cdef long total_words


cdef class _Parameters:
    cdef public int dim_size
    cdef public float init_alpha
    cdef public float min_alpha
    cdef public int window
    cdef public int negative
    cdef public float word_neg_power
    cdef public float entity_neg_power
    cdef public float sample
    cdef public int iteration
    cdef public int links_per_page
    cdef dict _kwargs

    def __init__(self, **kwargs):
        self._kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return self._kwargs


cdef _Parameters params


cdef class Wikipedia2Vec:
    cdef Dictionary _dictionary
    cdef list _train_history
    cdef public np.ndarray syn0
    cdef public np.ndarray syn1

    def __init__(self, dictionary):
        self._dictionary = dictionary
        self._train_history = []

    property dictionary:
        def __get__(self):
            return self._dictionary

    property train_history:
        def __get__(self):
            return self._train_history

    cpdef get_word(self, unicode word, default=None):
        return self._dictionary.get_word(word, default)

    cpdef get_entity(self, unicode title, bint resolve_redirect=True, default=None):
        return self._dictionary.get_entity(title, resolve_redirect, default)

    cpdef np.ndarray get_word_vector(self, unicode word):
        cdef Word obj

        obj = self._dictionary.get_word(word)
        if obj is None:
            raise KeyError()
        return self.syn0[obj.index]

    cpdef np.ndarray get_entity_vector(self, unicode title,
                                       bint resolve_redirect=True):
        cdef Entity obj

        obj = self._dictionary.get_entity(title, resolve_redirect=resolve_redirect)
        if obj is None:
            raise KeyError()
        return self.syn0[obj.index]

    cpdef np.ndarray get_vector(self, Item item):
        return self.syn0[item.index]

    cpdef list most_similar(self, Item item, int count=100):
        cdef np.ndarray vec

        vec = self.get_vector(item)
        return self.most_similar_by_vector(vec, count)

    cpdef list most_similar_by_vector(self, np.ndarray vec, int count=100):
        dst = (np.dot(self.syn0, vec) / np.linalg.norm(self.syn0, axis=1) / np.linalg.norm(vec))
        indexes = np.argsort(-dst)

        return [(self._dictionary.get_item_by_index(ind), dst[ind]) for ind in indexes[:count]]

    def init_sims(self):
        for i in xrange(self.syn0.shape[0]):
            self.syn0[i, :] /= np.sqrt((self.syn0[i, :] ** 2).sum(-1))

    def save(self, out_file):
        joblib.dump(dict(
            syn0=self.syn0,
            syn1=self.syn1,
            dictionary=self._dictionary.serialize(),
            train_history=self._train_history
        ), out_file)

    def save_word2vec_format(self, out_file):
        out_file.write('%d %d\n' % (self.syn0.shape[0], self.syn0.shape[1]))

        for item in sorted(self.dictionary, key=lambda o: o.doc_count, reverse=True):
            vec_str = ' '.join('%.4f' % v for v in self.get_vector(item))
            if isinstance(item, Word):
                text = item.text.encode('utf-8').replace(' ', '_')
            else:
                text = 'ENTITY/' + item.title.encode('utf-8').replace(' ', '_')

            out_file.write('%s %s\n' % (text, vec_str))

    @staticmethod
    def load(in_file, numpy_mmap_mode='c'):
        obj = joblib.load(in_file, mmap_mode=numpy_mmap_mode)
        if isinstance(obj['dictionary'], dict):
            dictionary = Dictionary.load(obj['dictionary'])
        else:
            dictionary = obj['dictionary']  # for backward compatibilit

        ret = Wikipedia2Vec(dictionary)
        ret.syn0 = obj['syn0']
        ret.syn1 = obj['syn1']
        ret._train_history = obj['train_history']

        return ret

    def train(self, dump_reader, link_graph_, pool_size, chunk_size, **kwargs):
        global dictionary, link_graph, word_counter, link_cursor, extractor, alpha, work, params, total_words

        start_time = time.time()

        params = _Parameters(**kwargs)

        words = list(self.dictionary.words())
        total_word_cnt = int(sum(w.count for w in words))
        total_words = total_word_cnt * params.iteration
        logger.info('Total number of word occurrence: %d', total_words)

        thresh = params.sample * total_word_cnt

        logger.info('Building a sampling table for frequent words...')

        sample_ints = multiprocessing.RawArray(c_uint32, len(words))
        for word in words:
            cnt = float(word.count)
            if params.sample == 0:
                word_prob = 1.0
            else:
                word_prob = min(1.0,
                                (np.sqrt(cnt / thresh) + 1) * (thresh / cnt))
            sample_ints[word.index] = int(round(word_prob * 2 ** 31))

        logger.info('Building tables for negative sampling...')

        word_neg_table = self._build_word_neg_table(params.word_neg_power)
        entity_neg_table = self._build_entity_neg_table(params.entity_neg_power)

        logger.info('Building tables for iterating nodes...')

        offset = self.dictionary.entity_offset
        indices = np.arange(offset,
                            offset + len(list(self.dictionary.entities())))
        if link_graph_:
            link_indices = multiprocessing.RawArray(
                c_uint32, np.random.permutation(indices)
            )
        else:
            link_indices = None

        link_cursor = multiprocessing.Value(c_uint32, 0)

        logger.info('Starting to train an embedding...')

        def iter_dump_reader():
            for n in range(params.iteration):
                logger.info('Iteration: %d', n)
                for page in dump_reader:
                    yield page

        vocab_size = len(self.dictionary)

        logger.info('Initializing weights...')
        dim_size = params.dim_size
        syn0_shared = multiprocessing.RawArray(c_float, dim_size * vocab_size)
        syn1_shared = multiprocessing.RawArray(c_float, dim_size * vocab_size)

        self.syn0 = np.frombuffer(syn0_shared, dtype=np.float32)
        self.syn0 = self.syn0.reshape(vocab_size, dim_size)

        self.syn1 = np.frombuffer(syn1_shared, dtype=np.float32)
        self.syn1 = self.syn1.reshape(vocab_size, dim_size)

        self.syn0[:] = (np.random.rand(vocab_size, dim_size) - 0.5) / dim_size
        self.syn1[:] = np.zeros((vocab_size, dim_size))

        dictionary = self.dictionary
        link_graph = link_graph_

        extractor = Extractor(dump_reader.language, dictionary.lowercase,
                              dictionary=dictionary)
        word_counter = multiprocessing.Value(c_uint64, 0)
        alpha = multiprocessing.RawValue(c_float, params.init_alpha)
        work = np.zeros(dim_size)

        init_args = (syn0_shared, syn1_shared, word_neg_table, entity_neg_table,
                     sample_ints, link_indices)

        with closing(Pool(pool_size, initializer=init_worker,
                          initargs=init_args)) as pool:
            for (n, _) in enumerate(pool.imap_unordered(
                train_page, iter_dump_reader(), chunksize=chunk_size
            )):
                if n % 10000 == 0:
                    prog = float(word_counter.value) / total_words
                    logger.info('Proccessing page #%d progress: %.1f%% '
                                'alpha: %.3f', n, prog * 100, alpha.value)

        logger.info('Terminated pool workers...')

        train_params = dict(
            dump_file=dump_reader.dump_file,
            train_time=time.time() - start_time,
        )
        train_params.update(params.to_dict())

        if link_graph is not None:
            train_params['link_graph'] = dict(
                build_params=link_graph.build_params
            )

        self._train_history.append(train_params)

    def _build_word_neg_table(self, power):
        items = list(self._dictionary.words())
        if power == 0:
            return self._build_uniform_neg_table(items)
        else:
            return self._build_unigram_neg_table(items, power)

    def _build_entity_neg_table(self, power):
        items = list(self._dictionary.entities())
        if power == 0:
            return self._build_uniform_neg_table(items)
        else:
            return self._build_unigram_neg_table(items, power)

    def _build_uniform_neg_table(self, items):
        return multiprocessing.RawArray(c_uint32, [o.index for o in items])

    def _build_unigram_neg_table(self, items, power, table_size=100000000):
        neg_table = multiprocessing.RawArray(c_uint32, table_size)
        items_pow = float(sum([item.count ** power for item in items]))

        index = 0
        cur = items[index].count ** power / items_pow

        for table_index in xrange(table_size):
            neg_table[table_index] = items[index].index
            if float(table_index) / table_size > cur:
                if index < len(items) - 1:
                    index += 1
                cur += items[index].count ** power / items_pow

        return neg_table


def init_worker(syn0_, syn1_, word_neg_table_, entity_neg_table_, sample_ints_,
                link_indices_):
    global syn0, syn1, extractor, sample_ints, word_neg_table,\
        entity_neg_table, link_indices

    syn0 = np.frombuffer(syn0_, dtype=np.float32)
    syn0 = syn0.reshape(len(dictionary), params.dim_size)
    syn1 = np.frombuffer(syn1_, dtype=np.float32)
    syn1 = syn1.reshape(len(dictionary), params.dim_size)

    word_neg_table = np.frombuffer(word_neg_table_, dtype=np.uint32)
    entity_neg_table = np.frombuffer(entity_neg_table_, dtype=np.uint32)

    sample_ints = np.frombuffer(sample_ints_, dtype=np.uint32)

    if link_indices_:
        link_indices = np.frombuffer(link_indices_, dtype=np.uint32)


def train_page(WikiPage page):
    cdef int pos, pos2, start, index, total_nodes, word_count, word, word2,\
        entity, entity2, start_node
    cdef tuple span
    cdef list words, target_words, entities, target_entities
    cdef WikiLink wiki_link

    # train using Wikipedia link graph
    if link_graph is not None:
        total_nodes = link_indices.shape[0]
        links_per_page = params.links_per_page

        with link_cursor.get_lock():
            start = link_cursor.value
            link_cursor.value = (start + links_per_page) % total_nodes

        for index in range(start, start + links_per_page):
            entity = link_indices[index % total_nodes]
            neighbors = link_graph.neighbor_indices(entity)
            for entity2 in neighbors:
                _train_pair(entity, entity2, alpha.value, params.negative, entity_neg_table)

    # train using Wikipedia words and anchors
    for paragraph in extractor.extract_paragraphs(page):
        words = [dictionary.get_word_index(w) for w in paragraph.words]
        word_count = 0
        for (pos, word) in enumerate(words):
            if word == -1:
                continue

            word_count += 1

            if sample_ints[word] < np.random.rand() * 2 ** 31:
                continue

            start = max(0, pos - params.window)
            target_words = words[start:pos + params.window + 1]
            for (pos2, word2) in enumerate(target_words, start):
                if word2 == -1 or pos2 == pos:
                    continue

                if sample_ints[word2] < np.random.rand() * 2 ** 31:
                    continue

                _train_pair(word, word2, alpha.value, params.negative, word_neg_table)

        entity2 = dictionary.get_entity_index(page.title)

        # train using word-entity co-occurrences
        for wiki_link in paragraph.wiki_links:
            entity = dictionary.get_entity_index(wiki_link.title)
            if entity == -1:
                continue

            span = wiki_link.span
            start = max(0, span[0] - params.window)
            target_words = words[start:span[1] + params.window]
            for (pos2, word2) in enumerate(target_words, start):
                if word2 == -1 or pos2 in range(span[0], span[1]):
                    continue

                if sample_ints[word2] < np.random.rand() * 2 ** 31:
                    continue

                _train_pair(entity, word2, alpha.value, params.negative, word_neg_table)

        with word_counter.get_lock():
            word_counter.value += word_count  # lock is required since += is not an atomic operation
            p = 1 - float(word_counter.value) / total_words
            alpha.value = max(params.min_alpha, params.init_alpha * p)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline _train_pair(
    int index1, int index2, np.float32_t alpha, int negative,
    np.ndarray[np.uint32_t, ndim=1] neg_table
):
    cdef np.float32_t label, f, g
    cdef int d, index
    cdef unsigned long row1, row2

    cdef np.float32_t *syn0_ = <np.float32_t *>(np.PyArray_DATA(syn0))
    cdef np.float32_t *syn1_ = <np.float32_t *>(np.PyArray_DATA(syn1))
    cdef np.float32_t *work_ = <np.float32_t *>(np.PyArray_DATA(work))

    memset(work_, 0, params.dim_size * cython.sizeof(np.float32_t))

    row1 = <unsigned long>index1 * params.dim_size
    for d in range(negative + 1):
        if d == 0:
            index = index2
            label = 1.0
        else:
            index = neg_table[np.random.randint(neg_table.shape[0])]
            if index == index2:
                continue
            label = 0.0

        row2 = <unsigned long>index * params.dim_size
        f = <np.float32_t>(blas.sdot(&params.dim_size, &syn0_[row1], &ONE,
                                     &syn1_[row2], &ONE))
        if f > MAX_EXP:
            g = (label - 1.0) * alpha
        elif f < -MAX_EXP:
            g = (label - 0.0) * alpha
        else:
            f = 1.0 / (1.0 + exp(-f))
            g = (label - f) * alpha

        blas.saxpy(&params.dim_size, &g, &syn1_[row2], &ONE, work_, &ONE)
        blas.saxpy(&params.dim_size, &g, &syn0_[row1], &ONE, &syn1_[row2], &ONE)

    blas.saxpy(&params.dim_size, &ONEF, work_, &ONE, &syn0_[row1], &ONE)
