# -*- coding: utf-8 -*-
# cython: profile=False

import joblib
import logging
import time
import six.moves.cPickle as pickle
import numpy as np
cimport cython
from contextlib import closing
from functools import partial
from multiprocessing.pool import Pool
from scipy.sparse import csr_matrix, lil_matrix
from tqdm import tqdm

from .dictionary cimport Dictionary, Entity
from .dump_db cimport DumpDB, Paragraph, WikiLink

logger = logging.getLogger(__name__)

cdef Dictionary _dictionary = None
cdef DumpDB _dump_db = None


cdef class LinkGraph:
    def __init__(self, Dictionary dictionary, np.ndarray indices, np.ndarray indptr,
                 dict build_params):
        self._dictionary = dictionary
        self._build_params = build_params
        self._indptr = indptr
        self._indices = indices

        self._offset = self._dictionary.entity_offset

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
    def load(in_file, dictionary, bint mmap=True):
        if mmap:
            obj = joblib.load(in_file, mmap_mode='r')
        else:
            obj = joblib.load(in_file)

        return LinkGraph(dictionary, obj['indices'], obj['indptr'], obj['build_params'])

    @staticmethod
    def build(dump_db, dictionary, pool_size, chunk_size, progressbar=True):
        global _dump_db, _dictionary

        _dump_db = dump_db
        _dictionary = dictionary

        start_time = time.time()

        logger.info('Step 1/2: Processing Wikipedia pages...')

        with closing(Pool(pool_size)) as pool:
            matrix = lil_matrix((dictionary.entity_size, dictionary.entity_size), dtype=np.bool)

            with tqdm(total=dump_db.page_size(), disable=not progressbar) as bar:
                f = partial(_process_page, offset=dictionary.entity_offset)

                for ret in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    if ret:
                        (page_index, dest_indices) = ret
                        matrix[page_index, dest_indices] = True
                        matrix[dest_indices, page_index] = True

                    bar.update(1)

        logger.info('Step 2/2: Converting matrix...')
        matrix = matrix.tocsr()
        matrix.indices += dictionary.entity_offset

        build_params = dict(dump_file=dump_db.dump_file, build_time=time.time() - start_time)

        return LinkGraph(dictionary, matrix.indices, matrix.indptr, build_params)


def _process_page(unicode title, int32_t offset):
    cdef int32_t page_index, dest_index
    cdef list dest_indices
    cdef Paragraph paragraph
    cdef WikiLink link

    page_index = _dictionary.get_entity_index(title)
    if page_index == -1:
        return None

    page_index -= offset

    dest_indices = []
    for paragraph in _dump_db.get_paragraphs(title):
        for link in paragraph.wiki_links:
            dest_index = _dictionary.get_entity_index(link.title)
            if dest_index != -1:
                dest_indices.append(dest_index - offset)

    return (page_index, dest_indices)
