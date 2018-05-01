# -*- coding: utf-8 -*-

import joblib
import logging
import six
import re
import time
from collections import defaultdict, Counter
from contextlib import closing
from functools import partial
from marisa_trie import Trie, RecordTrie
from multiprocessing.pool import Pool
from tqdm import tqdm
from uuid import uuid1

from .dump_db cimport DumpDB, Paragraph, WikiLink
from .utils.tokenizer.token cimport Token

logger = logging.getLogger(__name__)

cdef Dictionary _dictionary = None
cdef DumpDB _dump_db = None
cdef _tokenize_func = None
cdef _name_trie = None


cdef class Mention:
    def __init__(self, Dictionary dictionary, unicode text, int32_t start, int32_t end,
                 int32_t index, int32_t link_count, int32_t total_link_count, int32_t doc_count):
        self.text = text
        self.start = start
        self.end = end
        self.index = index
        self.link_count = link_count
        self.total_link_count = total_link_count
        self.doc_count = doc_count

        self._dictionary = dictionary

    @property
    def entity(self):
        return self._dictionary.get_entity_by_index(self.index)

    @property
    def link_prob(self):
        if self.doc_count > 0:
            return min(1.0, float(self.total_link_count) / self.doc_count)
        else:
            return 0.0

    @property
    def prior_prob(self):
        if self.total_link_count > 0:
            return min(1.0, float(self.link_count) / self.total_link_count)
        else:
            return 0.0

    def __repr__(self):
        if six.PY2:
            return b'<Mention %s -> %s>' % (self.text.encode('utf-8'), self.entity.title.encode('utf-8'))
        else:
            return '<Mention %s -> %s>' % (self.text, self.entity.title)


cdef class MentionDB(object):
    def __init__(self, mention_trie, dictionary, bint case_sensitive, int32_t max_mention_len,
                 dict build_params, unicode uuid):
        self.mention_trie = mention_trie
        self.uuid = uuid
        self.build_params = build_params

        self._dictionary = dictionary
        self._case_sensitive = case_sensitive
        self._max_mention_len = max_mention_len

    cpdef list prefix_search(self, unicode text, int32_t start=0):
        return sorted(self.mention_trie.prefixes(text[start:start+self._max_mention_len]), key=len,
                      reverse=True)

    cpdef list detect_mentions(self, unicode text, list tokens, set entity_indices_in_page=set()):
        cdef int32_t cur, start, end, index
        cdef unicode prefix, target_text
        cdef list ret, candidates
        cdef frozenset end_offsets
        cdef Mention mention
        cdef Token token

        end_offsets = frozenset([token.end for token in tokens])

        ret = []
        cur = 0

        if not self._case_sensitive:
            target_text = text.lower()

        for token in tokens:
            start = token.start
            if cur > start:
                continue

            for prefix in self.prefix_search(target_text, start=start):
                end = start + len(prefix)
                if end in end_offsets:
                    cur = end
                    candidates = self.mention_trie[prefix]
                    if len(candidates) == 1:
                        ret.append(Mention(self._dictionary, text[start:end], start, end,
                                           *candidates[0]))
                    else:
                        for (index, *args) in candidates:
                            if index in entity_indices_in_page:
                                ret.append(Mention(self._dictionary, text[start:end], start, end,
                                                   index, *args))
                                break
                    break

        return ret

    @staticmethod
    def build(dump_db, dictionary, tokenizer, min_link_prob, min_prior_prob, max_mention_len,
              case_sensitive, pool_size, chunk_size, progressbar=True):

        global _dictionary, _dump_db, _tokenize_func, _name_trie
        _dictionary = dictionary
        _dump_db = dump_db
        _tokenize_func = tokenizer

        start_time = time.time()

        logger.info('Step 1/3: Starting to iterate over Wikipedia pages...')

        name_dict = defaultdict(lambda: Counter())

        with closing(Pool(pool_size)) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_extract_links, max_mention_len=max_mention_len,
                            case_sensitive=case_sensitive)
                for ret in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    for (text, index) in ret:
                        name_dict[text][index] += 1
                    bar.update(1)

        logger.info('Step 2/3: Starting to count occurrences...')

        _name_trie = Trie(six.iterkeys(name_dict))
        name_counter = Counter()

        with closing(Pool(pool_size)) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_count_occurrences, max_mention_len=max_mention_len,
                            case_sensitive=case_sensitive)
                for names in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    name_counter.update(names)
                    bar.update(1)

        _name_trie = None

        logger.info('Step 3/3: Building DB...')

        def item_generator():
            for (name, entity_counter) in six.iteritems(name_dict):
                doc_count = name_counter[name]
                total_link_count = sum(entity_counter.values())

                if doc_count == 0:
                    continue

                link_prob = float(total_link_count) / doc_count
                if link_prob < min_link_prob:
                    continue

                for (index, link_count) in six.iteritems(entity_counter):
                    prior_prob = float(link_count) / total_link_count
                    if prior_prob < min_prior_prob:
                        continue

                    yield (name, (index, link_count, total_link_count, doc_count))

        mention_trie = RecordTrie('<IIII', item_generator())

        uuid = six.text_type(uuid1().hex)

        build_params = dict(dump_file=dump_db.dump_file,
                            dump_db=dump_db.uuid,
                            dictionary=dictionary.uuid,
                            build_time=time.time() - start_time)

        return MentionDB(mention_trie, dictionary, case_sensitive, max_mention_len, build_params,
                         uuid)

    def serialize(self):
        return dict(mention_trie=self.mention_trie.tobytes(),
                    kwargs=dict(max_mention_len=self._max_mention_len,
                                case_sensitive=self._case_sensitive,
                                build_params=self.build_params, uuid=self.uuid))

    def save(self, out_file):
        joblib.dump(self.serialize(), out_file)

    @staticmethod
    def load(target, dictionary):
        if not isinstance(target, dict):
            target = joblib.load(target)

        if target['kwargs']['build_params']['dictionary'] != dictionary.uuid:
            raise RuntimeError('The specified dictionary is different from the one used to build this DB')

        mention_trie = RecordTrie('<IIII')
        mention_trie = mention_trie.frombytes(target['mention_trie'])

        return MentionDB(mention_trie, dictionary, **target['kwargs'])


def _extract_links(unicode title, int32_t max_mention_len, bint case_sensitive):
    cdef unicode text
    cdef list ret
    cdef WikiLink wiki_link
    cdef Paragraph paragraph

    ret = []

    for paragraph in _dump_db.get_paragraphs(title):
        for wiki_link in paragraph.wiki_links:
            text = wiki_link.text

            if len(text) > max_mention_len:
                continue

            if not case_sensitive:
                text = text.lower()

            index = _dictionary.get_entity_index(wiki_link.title)
            if index != -1:
                ret.append((text, index))

    return ret


def _count_occurrences(unicode title, int32_t max_mention_len, bint case_sensitive):
    cdef int32_t start
    cdef unicode text, prefix
    cdef list ret
    cdef frozenset end_offsets
    cdef Token token
    cdef Paragraph paragraph

    ret = []

    for paragraph in _dump_db.get_paragraphs(title):
        text = paragraph.text
        tokens = _tokenize_func(text)

        if not case_sensitive:
            text = text.lower()

        end_offsets = frozenset(token.end for token in tokens)

        for token in tokens:
            start = token.start
            for prefix in _name_trie.prefixes(text[start:start+max_mention_len]):
                if (start + len(prefix)) in end_offsets:
                    ret.append(prefix)

    return frozenset(ret)
