# -*- coding: utf-8 -*-

import click
import collections
import logging
import time
import gzip
import cPickle as pickle
import numpy as np
cimport numpy as np
cimport cython
from contextlib import closing
from collections import deque
from functools import partial
from multiprocessing.pool import Pool
from scipy.sparse import lil_matrix

from dictionary cimport Dictionary, Entity

logger = logging.getLogger(__name__)


cdef class ClickGraph:
    def __init__(self, Dictionary dictionary, np.ndarray ind_ptr,
                 np.ndarray keys, np.ndarray probs, np.ndarray aliases,
                 dict build_params):
        self._dictionary = dictionary
        self._ind_ptr = ind_ptr
        self._keys = keys
        self._probs = probs
        self._aliases = aliases
        self._build_params = build_params

        self._offset = self._dictionary.entity_offset

    property build_params:
        def __get__(self):
            return self._build_params

    cpdef bint has_edge(self, src, dest):
        cdef int src_index, dest_index

        if isinstance(src, int):
            src_index = src
            dest_index = dest
        elif isinstance(src, Entity):
            src_index = src.index
            dest_index = dest.index
        elif isinstance(src, unicode):
            src_index = self._dictionary.get_entity_index(src)
            dest_index = self._dictionary.get_entity_index(dest)
        else:
            raise TypeError()

        src_index -= self._offset
        dest_index -= self._offset

        return dest_index in self._keys[self._ind_ptr[src_index]]

    cpdef list neighbors(self, item):
        cdef int index

        return [self._dictionary.get_entity_by_index(index)
                for index in self.neighbor_indices(item)]

    cpdef np.ndarray neighbor_indices(self, item):
        cdef int index

        if isinstance(item, int):
            index = item
        elif isinstance(item, Entity):
            index = item.index
        elif isinstance(item, unicode):
            index = self._dictionary.get_entity_index(item)
        else:
            raise TypeError()

        index = self._ind_ptr[index - self._offset]
        if index == -1:
            return np.array([])

        return self._keys[index][self._keys[index] != -1] + self._offset

    @cython.wraparound(False)
    cpdef list random_walk(self, item, int length=10,
                           bint return_indices=False):
        cdef int index, top_n, n, i, k
        cdef list ret
        cdef np.ndarray[np.float_t, ndim=1] floats
        cdef np.ndarray[np.int_t, ndim=1] ints
        cdef np.ndarray[np.int32_t, ndim=1] keys
        cdef np.ndarray[np.float32_t, ndim=1] probs
        cdef np.ndarray[np.int16_t, ndim=1] aliases

        if isinstance(item, int):
            index = item
        elif isinstance(item, Entity):
            index = item.index
        elif isinstance(item, unicode):
            index = self._dictionary.get_entity_index(item)
        else:
            raise TypeError()

        index = self._ind_ptr[index - self._offset]
        if index == -1:
            return []

        ret = []
        top_n = self._keys.shape[1]
        floats = np.random.sample(length)
        ints = np.random.randint(0, top_n, length)

        for n in range(length):
            keys = self._keys[index]
            probs = self._probs[index]
            aliases = self._aliases[index]

            i = ints[n]
            if floats[n] <= probs[i] or aliases[i] == -1:
                k = i
            else:
                k = aliases[i]

            ret.append(keys[k] + self._offset)

            index = self._ind_ptr[keys[k]]
            if index == -1:
                break

        if not return_indices:
            ret = [self._dictionary.get_entity_by_index(i) for i in ret]

        return ret

    def save(self, out_file):
        np.save(out_file + '_indptr.npy', self._ind_ptr)
        np.save(out_file + '_key.npy', self._keys)
        np.save(out_file + '_prob.npy', self._probs)
        np.save(out_file + '_alias.npy', self._aliases)
        with open(out_file + '_meta.pickle', 'w') as f:
            pickle.dump(dict(build_params=self._build_params), f)

    @staticmethod
    def load(in_file, dictionary, mmap=True):
        if mmap:
            ind_ptr = np.load(in_file + '_indptr.npy', mmap_mode='r')
            keys = np.load(in_file + '_key.npy', mmap_mode='r')
            probs = np.load(in_file + '_prob.npy', mmap_mode='r')
            aliases = np.load(in_file + '_alias.npy', mmap_mode='r')
        else:
            ind_ptr = np.load(in_file + '_indptr.npy')
            keys = np.load(in_file + '_index.npy')
            probs = np.load(in_file + '_prob.npy')
            aliases = np.load(in_file + '_alias.npy')

        with open(in_file + '_meta.pickle') as f:
            meta = pickle.load(f)

        return ClickGraph(dictionary, ind_ptr, keys, probs, aliases,
                          meta['build_params'])

    @staticmethod
    def build(clickstream_file, dictionary, top_n, pool_size, chunk_size):
        logger.info('Starting to build a Wikipedia click graph...')

        start_time = time.time()

        offset = dictionary.entity_offset
        num_entities = len(list(dictionary.entities()))

        if not isinstance(clickstream_file, collections.Iterable):
            clickstream_file = [clickstream_file]

        logger.info('Step 1/2: Loading dataset...')
        with closing(Pool(pool_size)) as pool:
            matrix = lil_matrix((num_entities, num_entities), dtype=np.int)

            for file_path in clickstream_file:
                with gzip.GzipFile(file_path) as f:
                    for (n, line) in enumerate(f):
                        if n == 0:
                            continue

                        line = line.rstrip().decode('utf-8')
                        (prev, curr, click_type, count) = line.split('\t')
                        if click_type != 'link':
                            continue

                        prev = prev.replace(u'_', u' ')
                        curr = curr.replace(u'_', u' ')

                        index1 = dictionary.get_entity_index(prev)
                        if index1 == -1:
                            continue

                        index2 = dictionary.get_entity_index(curr)
                        if index2 == -1:
                            continue

                        matrix[index1 - offset, index2 - offset] += int(count)

            matrix = matrix.tocsr()

            logger.info('Step 2/2: Constructing alias table...')

            ind_ptr = np.zeros(num_entities, dtype=np.int)
            keys = []
            probs = []
            aliases = []

            with click.progressbar(length=matrix.shape[0]) as bar:
                f = partial(_build_alias_entry, top_n=top_n)
                for (n, ret) in enumerate(pool.imap(f, matrix,
                                                    chunksize=chunk_size)):
                    bar.update(1)

                    if ret is None:
                        ind_ptr[n] = -1
                        continue

                    ind_ptr[n] = len(keys)

                    keys.append(ret[0])
                    probs.append(ret[1])
                    aliases.append(ret[2])

        del matrix

        keys = np.array(keys)
        probs = np.array(probs)
        aliases = np.array(aliases)

        build_params = dict(clickstream_file=clickstream_file,
                            build_time=time.time() - start_time)

        return ClickGraph(dictionary, ind_ptr, keys, probs, aliases,
                          build_params)


def _build_alias_entry(val, top_n):
    vec = val.toarray().flatten()
    nonzero = val.nonzero()[1]
    if nonzero.size == 0:
        return None

    keys = nonzero[np.argsort(-vec[nonzero])[:top_n]]
    weights = vec[keys]

    sum_weights = np.sum(weights)
    probs = (weights * (top_n / float(sum_weights))).astype(np.float32)

    key_len = keys.shape[0]
    if key_len < top_n:
        pad_len = top_n - key_len
        keys = np.concatenate((keys, -np.ones(pad_len, dtype=np.int32)))
        probs = np.concatenate((probs, np.zeros(pad_len, dtype=np.float32)))

    short_deque = deque(np.flatnonzero(probs < 1))
    long_deque = deque(np.flatnonzero(probs > 1))

    aliases = -np.ones(probs.shape[0], dtype=np.int16)

    while short_deque and long_deque:
        j = short_deque.pop()
        k = long_deque[-1]
        aliases[j] = k
        probs[k] -= 1.0 - probs[j]

        if probs[k] < 1.0:
            short_deque.append(k)
            long_deque.pop()

    return (keys, probs, aliases)
