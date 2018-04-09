# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
cimport cython
import joblib
import logging
import six
import tempfile
import time
import six.moves.cPickle as pickle
from collections import Counter
from contextlib import closing
from functools import partial
from marisa_trie import Trie
from multiprocessing.pool import Pool
from tqdm import tqdm
from uuid import uuid1

from .dump_db cimport DumpDB, Paragraph, WikiLink
from .utils.tokenizer import get_tokenizer
from .utils.tokenizer.base_tokenizer cimport BaseTokenizer
from .utils.tokenizer.token cimport Token

logger = logging.getLogger(__name__)

cdef DumpDB _dump_db = None
cdef BaseTokenizer _tokenizer = None
cdef _phrase_trie = None


cdef class PhraseDictionary:
    def __init__(self, phrase_trie, bint lowercase, dict build_params, unicode uuid=''):
        self.uuid = uuid
        self.phrase_trie = phrase_trie
        self.lowercase = lowercase
        self.build_params = build_params

    def __len__(self):
        return len(self.phrase_trie)

    def __iter__(self):
        for phrase in six.iterkeys(self.phrase_trie):
            yield phrase

    def __contains__(self, unicode key):
        return key in self.phrase_trie

    cpdef list keys(self):
        return self.phrase_trie.keys()

    @cython.wraparound(False)
    cpdef list prefix_search(self, unicode text, int32_t start=0, int32_t max_len=50):
        cdef list ret = self.phrase_trie.prefixes(text[start:start+max_len])
        ret.sort(key=len, reverse=True)
        return ret

    @staticmethod
    def build(dump_db, min_link_count, min_link_prob, lowercase, max_phrase_len, pool_size,
              chunk_size, progressbar=True):
        global _dump_db, _tokenizer

        start_time = time.time()

        _dump_db = dump_db
        _tokenizer = get_tokenizer(dump_db.language)

        phrase_counter = Counter()
        logger.info('Step 1/3: Counting anchor links...')

        with closing(Pool(pool_size)) as pool:
            f = partial(_extract_phrases, lowercase=lowercase, max_len=max_phrase_len)
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                for counter in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    phrase_counter.update(counter)
                    bar.update(1)

            phrase_counter = {w: c for (w, c) in six.iteritems(phrase_counter)
                              if c >= min_link_count}

            logger.info('Step 2/3: Counting occurrences...')

            with tempfile.NamedTemporaryFile() as f:
                Trie(six.iterkeys(phrase_counter)).save(f.name)

                occ_counter = Counter()

                f = partial(_count_occurrences, lowercase=lowercase, trie_file=f.name)
                with tqdm(total=dump_db.page_size(), disable=not progressbar) as bar:
                    for counter in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                        occ_counter.update(counter)
                        bar.update(1)

        logger.info('Step 3/3: Building TRIE...')
        phrase_trie = Trie(w for (w, c) in six.iteritems(occ_counter)
                           if float(phrase_counter[w]) / c >= min_link_prob)
        logger.info('%d phrases are successfully extracted', len(phrase_trie))

        build_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            min_link_count=min_link_count,
            min_link_prob=min_link_prob,
            build_time=time.time() - start_time,
        )

        uuid = six.text_type(uuid1().hex)

        return PhraseDictionary(phrase_trie, lowercase, build_params, uuid)

    def serialize(self):
        return dict(phrase_trie=self.phrase_trie.tobytes(),
                    kwargs=dict(uuid=self.uuid, lowercase=self.lowercase,
                                build_params=self.build_params))

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    @staticmethod
    def load(target):
        if not isinstance(target, dict):
            target = joblib.load(target)

        phrase_trie = Trie()
        phrase_trie.frombytes(target['phrase_trie'])

        return PhraseDictionary(phrase_trie=phrase_trie, **target['kwargs'])


def _extract_phrases(unicode title, bint lowercase, int max_len):
    cdef unicode text
    cdef Paragraph paragraph
    cdef WikiLink wiki_link

    ret = Counter()

    for paragraph in _dump_db.get_paragraphs(title):
        for wiki_link in paragraph.wiki_links:
            text = wiki_link.text
            if 1 < len(_tokenizer.tokenize(text)) <= max_len:
                if lowercase:
                    ret[text.lower()] += 1
                else:
                    ret[text] += 1

    return ret


def _count_occurrences(unicode title, trie_file, bint lowercase):
    global _phrase_trie

    cdef unicode text, prefix
    cdef list tokens
    cdef frozenset end_offsets
    cdef Token token
    cdef Paragraph paragraph

    if _phrase_trie is None:
        _phrase_trie = Trie()
        _phrase_trie.mmap(trie_file)

    ret = Counter()

    for paragraph in _dump_db.get_paragraphs(title):
        text = paragraph.text
        tokens = _tokenizer.tokenize(text)
        if lowercase:
            text = text.lower()
        end_offsets = frozenset(token.end for token in tokens)

        for token in tokens:
            for prefix in _phrase_trie.prefixes(text[token.start:]):
                if (token.start + len(prefix)) in end_offsets:
                    ret[prefix] += 1

    return ret
