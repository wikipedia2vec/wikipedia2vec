# -*- coding: utf-8 -*-
# cython: profile=False

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

from .extractor cimport Extractor
from .utils.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

_extractor = None
_tokenizer = None
_phrase_trie = None


cdef class PhraseDictionary(PrefixSearchable):
    def __init__(self, phrase_dict, bint lowercase, dict build_params):
        self._phrase_dict = phrase_dict
        self._lowercase = lowercase
        self._build_params = build_params

    property lowercase:
        def __get__(self):
            return self._lowercase

    property build_params:
        def __get__(self):
            return self._build_params

    def __len__(self):
        return len(self._phrase_dict)

    def __iter__(self):
        for phrase in six.iterkeys(self._phrase_dict):
            yield phrase

    def __contains__(self, key):
        return key in self._phrase_dict

    cpdef list keys(self):
        return self._phrase_dict.keys()

    cpdef list prefix_search(self, unicode text, int start=0):
        return sorted(self._phrase_dict.prefixes(text[start:]), key=len, reverse=True)

    @staticmethod
    def build(dump_reader, min_link_count, min_link_prob, lowercase, max_len, pool_size,
              chunk_size):
        global _extractor, _tokenizer

        start_time = time.time()

        _extractor = Extractor(dump_reader.language, lowercase)
        _tokenizer = get_tokenizer(dump_reader.language)

        phrase_counter = Counter()
        logger.info('Step 1/3: Counting anchor links...')

        with closing(Pool(pool_size)) as pool:
            f = partial(_extract_phrases, lowercase=lowercase, max_len=max_len)
            for counter in pool.imap_unordered(f, dump_reader, chunksize=chunk_size):
                phrase_counter.update(counter)

            phrase_counter = {w: c for (w, c) in six.iteritems(phrase_counter) if c >= min_link_count}

            logger.info('Step 2/3: Counting occurrences...')

            with tempfile.NamedTemporaryFile() as f:
                Trie(six.iterkeys(phrase_counter)).save(f.name)

                occ_counter = Counter()

                f = partial(_count_occurrences, lowercase=lowercase, trie_file=f.name)
                for counter in pool.imap_unordered(f, dump_reader, chunksize=chunk_size):
                    occ_counter.update(counter)

        logger.info('Step 3/3: Building TRIE...')
        phrase_dict = Trie(w for (w, c) in six.iteritems(occ_counter)
                           if float(phrase_counter[w]) / c >= min_link_prob)

        build_params = dict(
            dump_file=dump_reader.dump_file,
            min_link_count=min_link_count,
            min_link_prob=min_link_prob,
            build_time=time.time() - start_time,
        )

        return PhraseDictionary(phrase_dict, lowercase, build_params)

    def save(self, out_file):
        joblib.dump(dict(
            phrase_dict=self._phrase_dict.tobytes(),
            kwargs=dict(lowercase=self._lowercase, build_params=self._build_params),
        ), out_file)

    @staticmethod
    def load(in_file):
        obj = joblib.load(in_file)

        phrase_dict = Trie()
        phrase_dict.frombytes(obj['phrase_dict'])
        return PhraseDictionary(phrase_dict, **obj['kwargs'])


def _extract_phrases(page, lowercase, max_len):
    counter = Counter()
    try:
        for paragraph in _extractor.extract_paragraphs(page):
            for wiki_link in paragraph.wiki_links:
                text = wiki_link.text
                if 1 < len(_tokenizer.tokenize(text)) <= max_len:
                    if lowercase:
                        counter[text.lower()] += 1
                    else:
                        counter[text] += 1
    except:
        logging.exception('Unknown exception')
        raise

    return counter


def _count_occurrences(page, trie_file, lowercase):
    global _phrase_trie

    if _phrase_trie is None:
        _phrase_trie = Trie()
        _phrase_trie.mmap(trie_file)

    ret = Counter()
    try:
        for paragraph in _extractor.extract_paragraphs(page):
            tokens = _tokenizer.tokenize(paragraph.text)
            text = paragraph.text
            if lowercase:
                text = text.lower()
            end_offsets = frozenset(t.span[1] for t in tokens)

            for token in tokens:
                start = token.span[0]
                for prefix in _phrase_trie.prefixes(text[start:]):
                    if (start + len(prefix)) in end_offsets:
                        ret[prefix] += 1
    except:
        logging.exception('Unknown exception')
        raise

    return ret
