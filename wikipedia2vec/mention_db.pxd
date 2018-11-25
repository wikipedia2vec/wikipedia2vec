# -*- coding: utf-8 -*-
# License: Apache License 2.0

import numpy as np
cimport numpy as np
from libc.stdint cimport int32_t

from .dictionary cimport Dictionary, Entity

ctypedef np.float32_t float32_t


cdef class Mention:
    cdef readonly unicode text
    cdef readonly int32_t index
    cdef readonly int32_t link_count
    cdef readonly int32_t total_link_count
    cdef readonly int32_t doc_count
    cdef readonly int32_t start
    cdef readonly int32_t end

    cdef Dictionary _dictionary


cdef class MentionDB:
    cdef readonly mention_trie
    cdef readonly data_trie
    cdef readonly unicode uuid
    cdef readonly dict build_params
    cdef Dictionary _dictionary
    cdef bint _case_sensitive
    cdef int32_t _max_mention_len

    cpdef list query(self, unicode)
    cpdef list prefix_search(self, unicode, int32_t start=?)
    cdef inline list _prefix_search(self, unicode, int32_t start=?)
    cpdef list detect_mentions(self, unicode, list, set entity_indices_in_page=?)
