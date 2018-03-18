# -*- coding: utf-8 -*-

import logging
import time
import six.moves.cPickle as pickle
import numpy as np
cimport cython
from contextlib import closing
from multiprocessing.pool import Pool
from scipy.sparse import csr_matrix, lil_matrix

from .dictionary cimport Entity
from .extractor cimport Extractor

logger = logging.getLogger(__name__)

_extractor = None


cdef class LinkGraph:
    def __init__(self, Dictionary dictionary, np.ndarray indices, np.ndarray indptr,
                 dict build_params):
        self._dictionary = dictionary
        self._indices = indices
        self._indptr = indptr
        self._build_params = build_params
        self._offset = self._dictionary.entity_offset

    property build_params:
        def __get__(self):
            return self._build_params

    cpdef bint has_edge(self, item1, item2):
        cdef int index1, index2
        cdef np.ndarray[np.int32_t, ndim=1] indices

        if isinstance(item1, int):
            index1 = item1
            index2 = item2
        elif isinstance(item1, Entity):
            index1 = item1.index
            index2 = item2.index
        elif isinstance(item1, unicode):
            index1 = self._dictionary.get_entity_index(item1)
            index2 = self._dictionary.get_entity_index(item2)
        else:
            raise TypeError()

        index1 -= self._offset
        index2 -= self._offset

        indices = self._indices[self._indptr[index1]:self._indptr[index1 + 1]]
        return index2 in indices

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

        index -= self._offset

        return (self._indices[self._indptr[index]:self._indptr[index + 1]] + self._offset)

    @cython.wraparound(False)
    cpdef list random_walk(self, item, int length=10,
                           bint return_indices=False):
        cdef int index, neighbor
        cdef list ret
        cdef np.ndarray[np.int32_t, ndim=1] indices, indptr, neighbors

        indices = self._indices
        indptr = self._indptr

        if isinstance(item, int):
            index = item
        elif isinstance(item, Entity):
            index = item.index
        elif isinstance(item, unicode):
            index = self._dictionary.get_entity_index(item)
        else:
            raise TypeError()

        index -= self._offset
        ret = []
        for _ in range(length):
            neighbors = indices[indptr[index]:indptr[index + 1]]
            if neighbors.size == 0:
                break
            index = np.random.choice(neighbors)
            ret.append(index + self._offset)

        if not return_indices:
            ret = [self._dictionary.get_entity_by_index(index) for index in ret]

        return ret

    def save(self, out_file):
        np.save(out_file + '_indices.npy', self._indices)
        np.save(out_file + '_indptr.npy', self._indptr)
        with open(out_file + '_meta.pickle', 'wb') as f:
            pickle.dump(dict(build_params=self._build_params), f)

    @staticmethod
    def load(in_file, dictionary, mmap=True):
        if mmap:
            indices = np.load(in_file + '_indices.npy', mmap_mode='r')
            indptr = np.load(in_file + '_indptr.npy', mmap_mode='r')
        else:
            indices = np.load(in_file + '_indices.npy')
            indptr = np.load(in_file + '_indptr.npy')

        with open(in_file + '_meta.pickle', 'rb') as f:
            meta = pickle.load(f)

        return LinkGraph(dictionary, indices, indptr, meta['build_params'])

    @staticmethod
    def build(dump_reader, dictionary, pool_size, chunk_size):
        logger.info('Starting to build a Wikipedia link graph...')

        start_time = time.time()

        offset = dictionary.entity_offset
        num_entities = len(list(dictionary.entities()))

        logger.info('Step 1/2: Processing Wikipedia pages...')

        global _extractor
        _extractor = Extractor(dump_reader.language)

        with closing(Pool(pool_size)) as pool:
            matrix = lil_matrix((num_entities, num_entities), dtype=np.bool)

            for (page, paragraphs) in pool.imap_unordered(_extract_paragraphs, dump_reader,
                                                          chunksize=chunk_size):
                ind1 = dictionary.get_entity_index(page.title)
                if ind1 == -1:
                    continue
                ind1 -= offset

                if page.is_redirect:
                    continue

                for paragraph in paragraphs:
                    for wiki_link in paragraph.wiki_links:
                        ind2 = dictionary.get_entity_index(wiki_link.title)
                        if ind2 == -1:
                            continue
                        ind2 -= offset

                        matrix[ind1, ind2] = True
                        matrix[ind2, ind1] = True

        _extractor = None

        logger.info('Step 2/2: Converting matrix...')
        matrix = matrix.tocsr()

        build_params = dict(dump_file=dump_reader.dump_file, build_time=time.time() - start_time)

        return LinkGraph(dictionary, matrix.indices, matrix.indptr, build_params)


def _extract_paragraphs(page):
    try:
        return (page, _extractor.extract_paragraphs(page))
    except:
        logging.exception('Unknown exception')
        raise
