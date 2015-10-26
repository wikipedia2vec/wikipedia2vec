# -*- coding: utf-8 -*-
# cython: profile=False

import numpy as np
cimport numpy as np
cimport cython
from scipy.linalg cimport cython_blas as blas
from libc.math cimport exp
from libc.string cimport memset

from . import REAL
from dictionary cimport Dictionary, Item, Word, Entity
cimport wiki_page
from wiki_page cimport WikiPage

ctypedef np.float32_t REAL_t

cdef np.ndarray syn0 = None
cdef np.ndarray syn1 = None
cdef Dictionary dictionary = None
cdef int size

cdef int ONE = 1
cdef REAL_t ONEF = <REAL_t>1.0
cdef int MAX_EXP = 6

instance = None
syn0_shared = None
syn1_shared = None
word_neg_table = None
entity_neg_table = None
work = None


def init_worker(
    instance_, syn0_shared, syn1_shared, word_neg_table_, entity_neg_table_
):
    global instance, dictionary, size, work, syn0, syn1, word_neg_table,\
        entity_neg_table

    instance = instance_
    word_neg_table = word_neg_table_
    entity_neg_table = entity_neg_table_

    dictionary = instance.dictionary
    syn0 = np.frombuffer(syn0_shared, dtype=REAL)
    syn0 = syn0.reshape(len(dictionary), instance._size)
    syn1 = np.frombuffer(syn1_shared, dtype=REAL)
    syn1 = syn1.reshape(len(dictionary), instance._size)
    size = syn0.shape[1]
    work = np.zeros(size)


def train_page(WikiPage page):
    cdef int pos, pos2, start, window, word_count
    cdef Word word, word2
    cdef Entity entity, entity2
    cdef list paragraph, target_words, entity_list, word_list
    cdef tuple span

    for paragraph in page.extract_paragraphs(
        generate_links=instance._generate_links
    ):
        word_list = []
        entity_list = []
        for token in paragraph:
            if isinstance(token, wiki_page.Word):
                try:
                    word = dictionary.get_word(token.text.lower())
                    word_list.append(word)
                except KeyError:
                    word_list.append(None)

            elif isinstance(token, wiki_page.WikiLink):
                start = len(word_list)
                span = (start, start + len(token.words))
                try:
                    entity_list.append(
                        (dictionary.get_entity(token.title), span)
                    )
                except KeyError:
                    pass

                for w in token.words:
                    try:
                        word = dictionary.get_word(w.text.lower())
                        word_list.append(word)
                    except KeyError:
                        word_list.append(None)

            else:
                raise RuntimeError('Invalid token type')

        word_count = 0

        # train using word co-occurrences
        for (pos, word) in enumerate(word_list):
            if not word:
                continue
            word_count += 1

            window = instance._word_window
            start = max(0, pos - window)
            target_words = word_list[start:pos + window + 1]
            for (pos2, word2) in enumerate(target_words, start):
                if word2 and not (pos2 == pos):
                    _train_pair(
                        word, word2, instance._word_alpha.value,
                        instance._word_negative, word_neg_table
                    )

        try:
            entity2 = dictionary.get_entity(page.title)
        except KeyError:
            entity2 = None

        # train using word-entity co-occurrences
        for (entity, span) in entity_list:
            if not entity:
                continue

            window = instance._word_window
            start = max(0, span[0] - window)
            target_words = word_list[start:span[1] + window]
            for (pos2, word2) in enumerate(target_words, start):
                if word2 and pos2 not in range(span[0], span[1]):
                    _train_pair(
                        entity, word2, instance._word_alpha.value,
                        instance._word_negative, word_neg_table
                    )

            # train using entity links
            if entity2:
                _train_pair(
                    entity, entity2, instance._entity_alpha.value,
                    instance._entity_negative, entity_neg_table
                )

        with instance._word_counter.get_lock():
            instance._word_counter.value += word_count  # lock is required since += is not an atomic operation
            p = 1 - float(instance._word_counter.value) / instance._total_words

            instance._word_alpha.value = max(
                instance._word_min_alpha,
                instance._word_initial_alpha * p
            )
            instance._entity_alpha.value = max(
                instance._entity_min_alpha,
                instance._entity_initial_alpha * p
            )


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
cdef inline _train_pair(Item item, Item item2, REAL_t alpha, int negative,
                        neg_table):
    cdef REAL_t label, f, g
    cdef int d
    cdef unsigned long index, row1, row2

    cdef REAL_t *syn0_ = <REAL_t *>(np.PyArray_DATA(syn0))
    cdef REAL_t *syn1_ = <REAL_t *>(np.PyArray_DATA(syn1))
    cdef REAL_t *work_ = <REAL_t *>(np.PyArray_DATA(work))

    memset(work_, 0, size * cython.sizeof(REAL_t))

    index = item.index
    row1 = index * size
    for d in range(negative+1):
        if d == 0:
            index = item2.index
            label = 1.0
        else:
            index = neg_table[np.random.randint(len(neg_table))]
            if index == item2.index:
                continue
            label = 0.0

        row2 = index * size
        f = <REAL_t>(blas.sdot(&size, &syn0_[row1], &ONE, &syn1_[row2], &ONE))
        if f > MAX_EXP:
            g = (label - 1.0) * alpha
        if f < -MAX_EXP:
            g = (label - 0.0) * alpha
        else:
            f = 1.0 / (1.0 + exp(-f))
            g = (label - f) * alpha

        blas.saxpy(&size, &g, &syn1_[row2], &ONE, work_, &ONE)
        blas.saxpy(&size, &g, &syn0_[row1], &ONE, &syn1_[row2], &ONE)

    blas.saxpy(&size, &ONEF, work_, &ONE, &syn0_[row1], &ONE)
