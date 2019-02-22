# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals

import ctypes
import joblib
import logging
import math
import mmap
import multiprocessing
import numpy as np
import os
import pkg_resources
import random
import re
import time
import six
import six.moves.cPickle as pickle
import sys
import uuid
cimport cython
cimport numpy as np
np.import_array()
from cpython cimport array
from collections import defaultdict
from ctypes import c_float, c_int32
from contextlib import closing
from itertools import islice
from libc.math cimport exp
from libc.stdint cimport int32_t, uintptr_t
from libc.string cimport memset
from marisa_trie import Trie, RecordTrie
from multiprocessing.pool import Pool
from scipy.linalg cimport cython_blas as blas
from tqdm import tqdm

from .dictionary cimport Dictionary, Item, Word, Entity
from .dump_db cimport Paragraph, WikiLink, DumpDB
from .link_graph cimport LinkGraph
from .mention_db cimport MentionDB, Mention
from .utils.random cimport seed, randint_c
from .utils.sentence_detector.sentence cimport Sentence
from .utils.tokenizer import get_default_tokenizer
from .utils.tokenizer.token cimport Token

cdef int32_t INT32_MAX = np.iinfo(np.int32).max
ctypedef np.float32_t float32_t

DEF MAX_EXP = 6
DEF EXP_TABLE_SIZE = 1000

logger = logging.getLogger(__name__)


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
    cdef dict _params

    def __init__(self, params):
        self._params = params
        for key, value in params.items():
            setattr(self, key, value)

    def __reduce__(self):
        return (self.__class__, (self._params,))

    def to_dict(self):
        return self._params


cdef class Wikipedia2Vec:
    cdef Dictionary _dictionary
    cdef dict _train_params
    cdef public np.ndarray syn0
    cdef public np.ndarray syn1

    def __init__(self, dictionary):
        self._dictionary = dictionary

    @property
    def dictionary(self):
        return self._dictionary

    @property
    def train_params(self):
        return self._train_params

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

    cpdef list most_similar(self, Item item, count=100, min_count=None):
        cdef np.ndarray vec

        vec = self.get_vector(item)
        return self.most_similar_by_vector(vec, count, min_count=min_count)

    cpdef list most_similar_by_vector(self, np.ndarray vec, count=100, min_count=None):
        if min_count is None:
            min_count = 0

        counts = np.concatenate((
            self._dictionary._word_stats[:, 0],
            self._dictionary._entity_stats[:, 0]))
        dst = (np.dot(self.syn0, vec) / np.linalg.norm(self.syn0, axis=1) / np.linalg.norm(vec))
        dst[counts<min_count] = -100
        indexes = np.argsort(-dst)


        return [(self._dictionary.get_item_by_index(ind), dst[ind]) for ind in indexes[:count]]

    def save(self, out_file):
        joblib.dump(dict(
            syn0=self.syn0,
            syn1=self.syn1,
            dictionary=self._dictionary.serialize(),
            train_params=self._train_params
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
        ret._train_params = obj.get('train_params')

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

        dictionary = Dictionary(word_dict, entity_dict, redirect_dict, word_stats, entity_stats,
                                None, False, dict())
        ret = Wikipedia2Vec(dictionary)
        ret.syn0 = syn0
        ret.syn1 = None

        return ret

    def train(self, dump_db, link_graph, mention_db, tokenizer, sentence_detector, pool_size,
              chunk_size, progressbar=True, **kwargs):
        cdef float32_t [:, :] syn0_carr, syn1_carr

        start_time = time.time()

        params = _Parameters(kwargs)

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
            sample_ints[word.index] = int(round(word_prob * INT32_MAX))

        logger.info('Building tables for negative sampling...')

        word_neg_table = self._build_word_neg_table(params.word_neg_power)
        entity_neg_table = self._build_entity_neg_table(params.entity_neg_power)

        logger.info('Building tables for link indices...')

        total_page_count = dump_db.page_size() * params.iteration

        if link_graph is not None:
            offset = self.dictionary.entity_offset
            indices = np.arange(offset, offset + self.dictionary.entity_size)
            rep = int(math.ceil(float(total_page_count) * params.entities_per_page / indices.size))
            link_indices = multiprocessing.RawArray(c_int32,
                np.concatenate([np.random.permutation(indices) for _ in range(rep)])
            )
        else:
            link_indices = None

        logger.info('Starting to train embeddings...')

        exp_table = multiprocessing.RawArray(c_float, EXP_TABLE_SIZE)
        for i in range(EXP_TABLE_SIZE):
            exp_table[i] = <float32_t>exp((i / <float32_t>EXP_TABLE_SIZE * 2 - 1) * MAX_EXP)
            exp_table[i] = <float32_t>(exp_table[i] / (exp_table[i] + 1))

        link_graph_obj = None
        if link_graph is not None:
            link_graph_obj = link_graph.serialize(shared_array=True)

        mention_db_obj = None
        if mention_db is not None:
            mention_db_obj = mention_db.serialize()

        logger.info('Initializing weights...')

        dim_size = params.dim_size
        vocab_size = len(self.dictionary)

        if sys.platform == 'win32':
            syn0_addr = uuid.uuid1().hex
            syn0_mmap = mmap.mmap(-1, dim_size * vocab_size * ctypes.sizeof(c_float),
                                  tagname=syn0_addr)
            syn1_addr = uuid.uuid1().hex
            syn1_mmap = mmap.mmap(-1, dim_size * vocab_size * ctypes.sizeof(c_float),
                                  tagname=syn1_addr)
            self.syn0 = np.frombuffer(syn0_mmap, dtype=np.float32).reshape(vocab_size, dim_size)
            self.syn1 = np.frombuffer(syn1_mmap, dtype=np.float32).reshape(vocab_size, dim_size)

        else:
            syn0_mmap = mmap.mmap(-1, dim_size * vocab_size * ctypes.sizeof(c_float))
            syn1_mmap = mmap.mmap(-1, dim_size * vocab_size * ctypes.sizeof(c_float))
            self.syn0 = np.frombuffer(syn0_mmap, dtype=np.float32).reshape(vocab_size, dim_size)
            self.syn1 = np.frombuffer(syn1_mmap, dtype=np.float32).reshape(vocab_size, dim_size)
            syn0_carr = self.syn0
            syn1_carr = self.syn1
            syn0_addr = <uintptr_t>&syn0_carr[0, 0]
            syn1_addr = <uintptr_t>&syn1_carr[0, 0]

        self.syn0[:] = (np.random.rand(vocab_size, dim_size) - 0.5) / dim_size
        self.syn1[:] = np.zeros((vocab_size, dim_size))

        init_args = (
            dump_db,
            self.dictionary.serialize(shared_array=True),
            link_graph_obj,
            mention_db_obj,
            tokenizer,
            sentence_detector,
            syn0_addr,
            syn1_addr,
            word_neg_table,
            entity_neg_table,
            exp_table,
            sample_ints,
            link_indices,
            params
        )

        def args_generator(titles, iteration):
            random.shuffle(titles)
            for (n, title) in enumerate(titles, len(titles) * iteration):
                alpha = max(params.min_alpha,
                            params.init_alpha * (1.0 - float(n) / total_page_count))
                yield (n, title, alpha)

        with closing(Pool(pool_size, initializer=init_worker, initargs=init_args)) as pool:
            titles = list(dump_db.titles())
            for i in range(params.iteration):
                with tqdm(total=len(titles), mininterval=0.5, disable=not progressbar,
                          desc='Iteration %d/%d' % (i + 1, params.iteration)) as bar:
                    for _ in pool.imap_unordered(train_page, args_generator(titles, i),
                                                 chunksize=chunk_size):
                        bar.update(1)

            logger.info('Terminating pool workers...')

        self.syn0 = self.syn0.copy()
        syn0_mmap.close()
        self.syn1 = self.syn1.copy()
        syn1_mmap.close()

        train_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            dictionary=self.dictionary.uuid,
            tokenizer='%s.%s' % (tokenizer.__class__.__module__, tokenizer.__class__.__name__),
            train_time=time.time() - start_time,
            version=pkg_resources.get_distribution('wikipedia2vec').version,
        )
        train_params.update(params.to_dict())

        if link_graph is not None:
            train_params['link_graph'] = link_graph.uuid

        if mention_db is not None:
            train_params['mention_db'] = mention_db.uuid

        if sentence_detector is not None:
            train_params['sentence_detector'] = '%s.%s' % (sentence_detector.__class__.__module__,
                                                           sentence_detector.__class__.__name__)

        self._train_params = train_params

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


cdef DumpDB dump_db
cdef Dictionary dictionary
cdef LinkGraph link_graph
cdef MentionDB mention_db
cdef tokenizer
cdef sentence_detector
cdef float32_t [:, :] syn0
cdef float32_t [:, :] syn1
cdef int32_t [:] word_neg_table
cdef int32_t [:] entity_neg_table
cdef float32_t [:] exp_table
cdef int32_t [:] sample_ints
cdef int32_t [:] link_indices
cdef _Parameters params
cdef float32_t [:] work


def init_worker(dump_db_, dictionary_obj, link_graph_obj, mention_db_obj, tokenizer_,
                sentence_detector_,  syn0_addr, syn1_addr, word_neg_table_, entity_neg_table_,
                exp_table_, sample_ints_, link_indices_, params_):
    global dump_db, dictionary, link_graph, mention_db, tokenizer, sentence_detector, syn0, syn1,\
        word_neg_table, entity_neg_table, exp_table, sample_ints, link_indices, params, work

    cdef uintptr_t syn0_ptr, syn1_ptr

    dump_db = dump_db_
    tokenizer = tokenizer_
    sentence_detector = sentence_detector_
    word_neg_table = word_neg_table_
    entity_neg_table = entity_neg_table_
    exp_table = exp_table_
    sample_ints = sample_ints_
    link_indices = link_indices_
    params = params_

    np.random.seed()
    seed(np.random.randint(2 ** 31))

    dictionary = Dictionary.load(dictionary_obj)

    if link_graph_obj is not None:
        link_graph = LinkGraph.load(link_graph_obj, dictionary)
    else:
        link_graph = None

    if mention_db_obj is not None:
        mention_db = MentionDB.load(mention_db_obj, dictionary)
    else:
        mention_db = None

    vocab_size = len(dictionary)
    if sys.platform == 'win32':
        syn0_mmap = mmap.mmap(-1, params.dim_size * vocab_size * ctypes.sizeof(c_float), tagname=syn0_addr)
        syn1_mmap = mmap.mmap(-1, params.dim_size * vocab_size * ctypes.sizeof(c_float), tagname=syn1_addr)
        syn0 = np.frombuffer(syn0_mmap, dtype=np.float32).reshape(-1, params.dim_size)
        syn1 = np.frombuffer(syn1_mmap, dtype=np.float32).reshape(-1, params.dim_size)

    else:
        syn0_ptr = syn0_addr
        syn1_ptr = syn1_addr
        syn0 = np.PyArray_SimpleNewFromData(2, [vocab_size, params.dim_size], np.NPY_FLOAT32,
                                            <float32_t *>syn0_ptr)
        syn1 = np.PyArray_SimpleNewFromData(2, [vocab_size, params.dim_size], np.NPY_FLOAT32,
                                            <float32_t *>syn1_ptr)
    work = np.zeros(params.dim_size, dtype=np.float32)


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
def train_page(tuple arg):
    cdef bint matched
    cdef int32_t i, j, n, start, end, span_start, span_end, ent_start, ent_end, word, word2,\
        entity, text_len, token_len, window
    cdef int32_t [:] words, word_pos, sent_char_pos, sent_token_pos
    cdef char [:] link_flags
    cdef const int32_t [:] neighbors
    cdef unicode text, title
    cdef list tokens, paragraphs, target_links
    cdef set entity_indices
    cdef Token token
    cdef Sentence sentence
    cdef WikiLink wiki_link
    cdef Mention mention
    cdef float32_t alpha_

    (n, title, alpha_) = arg

    # train using Wikipedia link graph
    if link_graph is not None:
        start = n * params.entities_per_page
        for i in range(start, start + params.entities_per_page):
            entity = link_indices[i]
            neighbors = link_graph.neighbor_indices(entity)
            for j in range(len(neighbors)):
                _train_pair(entity, neighbors[j], alpha_, params.negative, entity_neg_table)

    paragraphs = dump_db.get_paragraphs(title)
    if mention_db is not None:
        entity_indices = set()
        entity_indices.add(dictionary.get_entity_index(title))

        for paragraph in paragraphs:
            for wiki_link in paragraph.wiki_links:
                entity_indices.add(dictionary.get_entity_index(wiki_link.title))

        entity_indices.discard(-1)

    # train using Wikipedia words and anchors
    for paragraph in paragraphs:
        text = paragraph.text
        text_len = len(text)
        tokens = tokenizer.tokenize(text)
        token_len = len(tokens)
        if not tokens or token_len < dictionary.min_paragraph_len:
            continue

        words = cython.view.array(shape=(token_len,), itemsize=sizeof(int32_t), format='i')
        word_pos = cython.view.array(shape=(text_len + 1,), itemsize=sizeof(int32_t), format='i')

        if sentence_detector is not None:
            sent_char_pos = cython.view.array(shape=(text_len + 1,), itemsize=sizeof(int32_t),
                                              format='i')
            sent_token_pos = cython.view.array(shape=(token_len,), itemsize=sizeof(int32_t),
                                               format='i')

            memset(&sent_char_pos[0], 0, (text_len + 1) * cython.sizeof(int32_t))
            for (i, sentence) in enumerate(sentence_detector.detect_sentences(text)):
                sent_char_pos[sentence.start:sentence.end] = i

        j = 0
        for (i, token) in enumerate(tokens):
            if dictionary.lowercase:
                words[i] = dictionary.get_word_index(token.text.lower())
            else:
                words[i] = dictionary.get_word_index(token.text)

            if sentence_detector is not None:
                sent_token_pos[i] = sent_char_pos[token.start]

            if i > 0:
                word_pos[j:token.start] = i - 1
                j = token.start
        word_pos[j:] = i

        for i in range(len(words)):
            word = words[i]
            if word == -1:
                continue

            if sample_ints[word] < randint_c():
                continue

            window = randint_c() % params.window + 1
            start = max(0, i - window)
            end = min(len(words), i + window + 1)
            for j in range(start, end):
                word2 = words[j]

                if word2 == -1 or i == j:
                    continue

                if sample_ints[word2] < randint_c():
                    continue

                if sentence_detector is not None and sent_token_pos[i] != sent_token_pos[j]:
                    continue

                _train_pair(word, word2, alpha_, params.negative, word_neg_table)

        link_flags = cython.view.array(shape=(text_len + 1,), itemsize=sizeof(char), format='c')
        link_flags[:] = 0

        target_links = []

        for wiki_link in paragraph.wiki_links:
            entity = dictionary.get_entity_index(wiki_link.title)
            if entity == -1:
                continue

            if not (0 <= wiki_link.start <= text_len and 0 <= wiki_link.end <= text_len):
                logger.warn('Detected invalid span of a wiki link')
                continue

            target_links.append((entity, wiki_link.start, wiki_link.end))

            link_flags[wiki_link.start:wiki_link.end] = 1

        if mention_db is not None:
            for mention in mention_db.detect_mentions(text, tokens, entity_indices):
                matched = False
                for i in range(mention.start, mention.end):
                    if link_flags[i] == 1:
                        matched = True
                        break

                if not matched:
                    target_links.append((mention.index, mention.start, mention.end))

        for (entity, ent_start, ent_end) in target_links:
            span_start = word_pos[ent_start]
            span_end = word_pos[max(0, ent_end - 1)] + 1

            window = randint_c() % params.window + 1
            start = max(0, span_start - window)
            end = min(len(words), span_end + window)
            for j in range(start, end):
                word2 = words[j]
                if word2 == -1 or span_start <= j < span_end:
                    continue

                if sample_ints[word2] < randint_c():
                    continue

                if sentence_detector is not None and sent_char_pos[ent_start] != sent_token_pos[j]:
                    continue

                _train_pair(entity, word2, alpha_, params.negative, word_neg_table)
                _train_pair(word2, entity, alpha_, params.negative, entity_neg_table)


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
            neg_index = randint_c() % neg_table_size
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
