# -*- coding: utf-8 -*-
# cython: profile=False

import joblib
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
        self._build_params = build_params
        self._indptr = indptr

        self._offset = self._dictionary.entity_offset
        indices = indices + self._offset
        self._indices = indices

    property build_params:
        def __get__(self):
            return self._build_params

    cpdef list neighbors(self, Entity item):
        return [self._dictionary.get_entity_by_index(i) for i in self.neighbor_indices(item.index)]

    @cython.boundscheck(False)
    @cython.initializedcheck(False)
    @cython.wraparound(False)
    cdef const int32_t [:] neighbor_indices(self, int32_t index) nogil:
        index -= self._offset
        return self._indices[self._indptr[index]:self._indptr[index + 1]]

    def save(self, out_file):
        joblib.dump(dict(indices=np.asarray(self._indices, dtype=np.int32),
                         indptr=np.asarray(self._indptr, dtype=np.int32),
                         build_params=self._build_params), out_file)

    @staticmethod
    def load(in_file, dictionary, mmap=True):
        if mmap:
            obj = joblib.load(in_file, mmap_mode='r')
        else:
            obj = joblib.load(in_file)

        return LinkGraph(dictionary, obj['indices'], obj['indptr'], obj['build_params'])

    @staticmethod
    def build(dump_reader, dictionary, pool_size, chunk_size):
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
