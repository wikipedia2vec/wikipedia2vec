# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals

import joblib
import logging
import multiprocessing
import numpy as np
import os
import random
import re
import time
import six
import six.moves.cPickle as pickle
cimport numpy as np
cimport cython
from cpython cimport array
from collections import defaultdict
from ctypes import c_float, c_int32
from contextlib import closing
from itertools import islice
from libc.math cimport exp
from libc.stdint cimport int32_t
from libc.stdlib cimport rand, RAND_MAX
from libc.string cimport memset
from marisa_trie import Trie, RecordTrie
from multiprocessing.pool import Pool
from scipy.linalg cimport cython_blas as blas
from tqdm import tqdm

from .dictionary cimport Dictionary, Item, Word, Entity
from .dump_db cimport Paragraph, WikiLink, DumpDB
from .link_graph cimport LinkGraph
from .utils.tokenizer.base_tokenizer cimport BaseTokenizer
from .utils.tokenizer.token cimport Token

ctypedef np.float32_t float32_t

DEF MAX_EXP = 6
DEF EXP_TABLE_SIZE = 1000

logger = logging.getLogger(__name__)

cdef Dictionary dictionary
cdef DumpDB dump_db
cdef LinkGraph link_graph
cdef BaseTokenizer tokenizer
cdef float32_t [:, :] syn0
cdef float32_t [:, :] syn1
cdef float32_t [:] work
cdef int32_t [:] word_neg_table
cdef int32_t [:] entity_neg_table
cdef float32_t [:] exp_table
cdef int32_t [:] sample_ints
cdef int32_t [:] link_indices
cdef float32_t total_page_count
cdef unicode language
cdef object alpha
cdef object link_cursor


cdef class _Parameters:
    cdef public int32_t dim_size
    cdef public float32_t init_alpha
    cdef public float32_t min_alpha
    cdef public int32_t window
    cdef public int32_t negative
    cdef public float32_t word_neg_power
    cdef public float32_t entity_neg_power
    cdef public float32_t sample
    cdef public int32_t iteration
    cdef public int32_t entities_per_page
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

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def train_history(self):
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

    cpdef np.ndarray get_entity_vector(self, unicode title, bint resolve_redirect=True):
        cdef Entity obj

        obj = self._dictionary.get_entity(title, resolve_redirect=resolve_redirect)
        if obj is None:
            raise KeyError()
        return self.syn0[obj.index]

    cpdef np.ndarray get_vector(self, Item item):
        return self.syn0[item.index]

    cpdef list most_similar(self, Item item, count=100):
        cdef np.ndarray vec

        vec = self.get_vector(item)
        return self.most_similar_by_vector(vec, count)

    cpdef list most_similar_by_vector(self, np.ndarray vec, count=100):
        dst = (np.dot(self.syn0, vec) / np.linalg.norm(self.syn0, axis=1) / np.linalg.norm(vec))
        indexes = np.argsort(-dst)

        return [(self._dictionary.get_item_by_index(ind), dst[ind]) for ind in indexes[:count]]

    def save(self, out_file):
        joblib.dump(dict(
            syn0=self.syn0,
            syn1=self.syn1,
            dictionary=self._dictionary.serialize(),
            train_history=self._train_history
        ), out_file)

    def save_text(self, out_file, out_format='default'):
        with open(out_file, 'wb') as f:
            if out_format == 'word2vec':
                f.write(('%d %d\n' % (len(self.dictionary), len(self.syn0[0]))).encode('utf-8'))

            for item in sorted(self.dictionary, key=lambda o: o.doc_count, reverse=True):
                vec_str = ' '.join('%.4f' % v for v in self.get_vector(item))
                if isinstance(item, Word):
                    text = item.text.replace('\t', ' ')
                else:
                    text = 'ENTITY/' + item.title.replace('\t', ' ')

                if out_format in ('word2vec', 'glove'):
                    text = text.replace(' ', '_')
                    f.write(('%s %s\n' % (text, vec_str)).encode('utf-8'))
                else:
                    f.write(('%s\t%s\n' % (text, vec_str)).encode('utf-8'))

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

    @staticmethod
    def load_text(in_file):
        words = defaultdict(int)
        entities = defaultdict(int)
        vectors = []

        with open(in_file, 'rb') as f:
            if b'\t' in list(islice(f, 2))[1]:
                sep = '\t'
            else:
                sep = ' '

            f.seek(0)
            n = 0
            for (i, line) in enumerate(f):
                line = line.decode('utf-8').rstrip()
                if i == 0 and re.match(r'^\d+\s\d+$', line):  # word2vec format
                    continue

                if sep == '\t':
                    (item_str, vec_str) = line.split(sep)
                    vectors.append(np.array([float(s) for s in vec_str.split(' ')], dtype=np.float32))
                else:
                    items = line.split(sep)
                    item_str = items[0].replace('_', ' ')
                    vectors.append(np.array([float(s) for s in items[1:]], dtype=np.float32))

                if item_str.startswith('ENTITY/'):
                    entities[item_str[7:]] = n
                else:
                    words[item_str] = n
                n += 1

        syn0 = np.empty((len(vectors), vectors[0].size))

        word_dict = Trie(words.keys())
        entity_dict = Trie(entities.keys())
        redirect_dict = RecordTrie('<I')

        for (word, ind) in six.iteritems(word_dict):
            syn0[ind] = vectors[words[word]]

        entity_offset = len(word_dict)
        for (title, ind) in six.iteritems(entity_dict):
            syn0[ind + entity_offset] = vectors[entities[title]]

        word_stats = np.zeros((len(word_dict), 2), dtype=np.int32)
        entity_stats = np.zeros((len(entity_dict), 2), dtype=np.int32)

        dictionary = Dictionary(word_dict, entity_dict, redirect_dict, None, word_stats,
                                entity_stats, None, False, dict())
        ret = Wikipedia2Vec(dictionary)
        ret.syn0 = syn0
        ret.syn1 = None
        ret._train_history = []

        return ret

    def train(self, dump_db_, link_graph_, pool_size, chunk_size, progressbar=True, **kwargs):
        global dictionary, dump_db, link_graph, tokenizer, syn0, syn1, work, word_neg_table,\
            entity_neg_table, exp_table, sample_ints, link_indices, link_cursor, alpha, params,\
            total_page_count

        start_time = time.time()

        params = _Parameters(**kwargs)

        words = list(self.dictionary.words())
        total_word_count = int(sum(w.count for w in words))
        logger.info('Total number of word occurrences: %d', total_word_count * params.iteration)

        thresh = params.sample * total_word_count

        logger.info('Building a sampling table for frequent words...')

        sample_ints = multiprocessing.RawArray(c_int32, len(words))
        for word in words:
            cnt = float(word.count)
            if params.sample == 0:
                word_prob = 1.0
            else:
                word_prob = min(1.0, (np.sqrt(cnt / thresh) + 1) * (thresh / cnt))
            sample_ints[word.index] = int(round(word_prob * RAND_MAX))

        logger.info('Building tables for negative sampling...')

        word_neg_table = self._build_word_neg_table(params.word_neg_power)
        entity_neg_table = self._build_entity_neg_table(params.entity_neg_power)

        logger.info('Building tables for link indices...')

        offset = self.dictionary.entity_offset
        indices = np.arange(offset, offset + len(list(self.dictionary.entities())))
        if link_graph_:
            link_indices = multiprocessing.RawArray(c_int32, np.random.permutation(indices))
        else:
            link_indices = None

        link_cursor = multiprocessing.Value(c_int32, 0)

        logger.info('Starting to train embeddings...')

        vocab_size = len(self.dictionary)

        logger.info('Initializing weights...')

        dim_size = params.dim_size
        syn0_shared = multiprocessing.RawArray(c_float, dim_size * vocab_size)
        syn1_shared = multiprocessing.RawArray(c_float, dim_size * vocab_size)

        self.syn0 = np.frombuffer(syn0_shared, dtype=np.float32)
        syn0 = self.syn0 = self.syn0.reshape(vocab_size, dim_size)

        self.syn1 = np.frombuffer(syn1_shared, dtype=np.float32)
        syn1 = self.syn1 = self.syn1.reshape(vocab_size, dim_size)

        self.syn0[:] = (np.random.rand(vocab_size, dim_size) - 0.5) / dim_size
        self.syn1[:] = np.zeros((vocab_size, dim_size))

        dump_db = dump_db_
        dictionary = self.dictionary
        link_graph = link_graph_
        tokenizer = dictionary.get_tokenizer()

        total_page_count = dump_db.page_size() * params.iteration
        alpha = multiprocessing.RawValue(c_float, params.init_alpha)
        work = np.zeros(dim_size, dtype=np.float32)

        exp_table = multiprocessing.RawArray(c_float, EXP_TABLE_SIZE)
        for i in range(EXP_TABLE_SIZE):
            exp_table[i] = <float32_t>exp((i / <float32_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
            exp_table[i] = <float32_t>(exp_table[i] / (exp_table[i] + 1))

        with closing(Pool(pool_size)) as pool:
            titles = list(dump_db.titles())
            for i in range(params.iteration):
                random.shuffle(titles)
                with tqdm(total=len(titles), mininterval=0.5, disable=not progressbar,
                          desc='Iteration %d/%d' % (i + 1, params.iteration)) as bar:
                    for _ in pool.imap_unordered(train_page, enumerate(titles, len(titles) * i),
                                                 chunksize=chunk_size):
                        bar.update(1)

            logger.info('Terminating pool workers...')

        train_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            dictionary=dictionary.uuid,
            train_time=time.time() - start_time,
        )
        train_params.update(params.to_dict())

        if link_graph is not None:
            train_params['link_graph'] = link_graph.uuid

        self._train_history.append(train_params)

    def _build_word_neg_table(self, float32_t power):
        if power == 0:
            return self._build_uniform_neg_table(self._dictionary.words())
        else:
            return self._build_unigram_neg_table(self._dictionary.words(), power)

    def _build_entity_neg_table(self, float32_t power):
        if power == 0:
            return self._build_uniform_neg_table(self._dictionary.entities())
        else:
            return self._build_unigram_neg_table(self._dictionary.entities(), power)

    def _build_uniform_neg_table(self, items):
        cdef Item item

        return multiprocessing.RawArray(c_int32, [item.index for item in items])

    def _build_unigram_neg_table(self, items, float32_t power, int32_t table_size=100000000):
        cdef int32_t index, table_index
        cdef float32_t cur, items_pow
        cdef Item item

        items = list(items)
        neg_table = multiprocessing.RawArray(c_int32, table_size)
        items_pow = float(sum([item.count ** power for item in items]))

        index = 0
        cur = items[index].count ** power / items_pow

        for table_index in range(table_size):
            neg_table[table_index] = items[index].index
            if float(table_index) / table_size > cur:
                if index < len(items) - 1:
                    index += 1
                cur += items[index].count ** power / items_pow

        return neg_table


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def train_page(tuple arg):
    cdef int32_t i, j, n, start, end, span_start, span_end, word, word2, entity, text_len,\
        token_len, total_nodes, window
    cdef int32_t [:] words, word_pos
    cdef const int32_t [:] neighbors
    cdef unicode text, title
    cdef list tokens
    cdef Token token
    cdef WikiLink wiki_link
    cdef float32_t alpha_

    (n, title) = arg

    alpha_ = alpha.value

    # train using Wikipedia link graph
    if link_graph is not None:
        total_nodes = link_indices.size

        with link_cursor.get_lock():
            start = link_cursor.value
            link_cursor.value = (start + params.entities_per_page) % total_nodes

        for i in range(start, start + params.entities_per_page):
            entity = link_indices[i % total_nodes]
            neighbors = link_graph.neighbor_indices(entity)
            for j in range(len(neighbors)):
                _train_pair(entity, neighbors[j], alpha_, params.negative, entity_neg_table)

    # train using Wikipedia words and anchors
    for paragraph in dump_db.get_paragraphs(title):
        text = paragraph.text
        text_len = len(text)
        tokens = tokenizer.tokenize(text)
        token_len = len(tokens)
        if not tokens or token_len < dictionary.min_paragraph_len:
            continue

        words = cython.view.array(shape=(len(tokens),), itemsize=sizeof(int32_t), format='i')
        word_pos = cython.view.array(shape=(text_len + 1,), itemsize=sizeof(int32_t), format='i')
        j = 0
        for (i, token) in enumerate(tokens):
            if dictionary.lowercase:
                words[i] = dictionary.get_word_index(token.text.lower())
            else:
                words[i] = dictionary.get_word_index(token.text)
            if i > 0:
                word_pos[j:token.start] = i - 1
                j = token.start
        word_pos[j:] = i

        for i in range(len(words)):
            word = words[i]
            if word == -1:
                continue

            if sample_ints[word] < rand():
                continue

            window = rand() % params.window + 1
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                word2 = words[j]

                if word2 == -1 or i == j:
                    continue

                if sample_ints[word2] < rand():
                    continue

                _train_pair(word, word2, alpha_, params.negative, word_neg_table)

        # train using word-entity co-occurrences
        for wiki_link in paragraph.wiki_links:
            entity = dictionary.get_entity_index(wiki_link.title)
            if entity == -1:
                continue

            if not (0 <= wiki_link.start <= text_len and 0 <= wiki_link.end <= text_len):
                logger.warn('Detected invalid span of a wiki link')
                continue

            span_start = word_pos[wiki_link.start]
            span_end = word_pos[max(0, wiki_link.end - 1)] + 1

            window = rand() % params.window + 1
            start = max(0, span_start - window)
            end = min(len(words), span_end + window)
            for j in range(start, end):
                word2 = words[j]
                if word2 == -1 or span_start <= j < span_end:
                    continue

                if sample_ints[word2] < rand():
                    continue

                _train_pair(entity, word2, alpha_, params.negative, word_neg_table)
                _train_pair(word2, entity, alpha_, params.negative, entity_neg_table)

    alpha.value = max(params.min_alpha, params.init_alpha * (1.0 - n / total_page_count))


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline void _train_pair(int32_t index1, int32_t index2, float32_t alpha, int32_t negative,
                             int32_t [:] neg_table) nogil:
    cdef float32_t label, f, g, f_dot
    cdef int32_t index, neg_index, d

    cdef int one = 1
    cdef int dim_size = params.dim_size
    cdef float32_t onef = <float32_t>1.0
    cdef int32_t neg_table_size = len(neg_table)

    memset(&work[0], 0, dim_size * cython.sizeof(float32_t))

    for d in range(negative + 1):
        if d == 0:
            index = index2
            label = 1.0
        else:
            neg_index = rand() % neg_table_size
            index = neg_table[neg_index]
            if index == index2:
                continue
            label = 0.0

        f_dot = <float32_t>(blas.sdot(&dim_size, &syn0[index1, 0], &one, &syn1[index, 0], &one))
        if f_dot >= MAX_EXP or f_dot <= -MAX_EXP:
            continue
        f = exp_table[<int>((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]
        g = (label - f) * alpha

        blas.saxpy(&dim_size, &g, &syn1[index, 0], &one, &work[0], &one)
        blas.saxpy(&dim_size, &g, &syn0[index1, 0], &one, &syn1[index, 0], &one)

    blas.saxpy(&dim_size, &onef, &work[0], &one, &syn0[index1, 0], &one)
