import logging
import time
from collections import Counter, defaultdict
from contextlib import closing
from functools import partial
from multiprocessing.pool import Pool
from typing import FrozenSet, Iterator, List, Tuple, Union
from uuid import uuid1

import cython
import joblib
from marisa_trie import RecordTrie, Trie
from tqdm import tqdm

from .dictionary import Dictionary, Entity
from .dump_db import DumpDB
from .utils.tokenizer.base_tokenizer import BaseTokenizer
from .utils.tokenizer.token import Token

logger = logging.getLogger(__name__)


@cython.cclass
class Mention:
    def __init__(
        self,
        dictionary: Dictionary,
        text: str,
        index: int,
        link_count: int,
        total_link_count: int,
        doc_count: int,
        start: int = -1,
        end: int = -1,
    ):
        self.text = text
        self.index = index
        self.link_count = link_count
        self.total_link_count = total_link_count
        self.doc_count = doc_count
        self.start = start
        self.end = end

        self._dictionary = dictionary

    @property
    def entity(self) -> Entity:
        return self._dictionary.get_entity_by_index(self.index)

    @property
    def link_prob(self) -> float:
        if self.doc_count > 0:
            return min(1.0, float(self.total_link_count) / self.doc_count)
        else:
            return 0.0

    @property
    def prior_prob(self) -> float:
        if self.total_link_count > 0:
            return min(1.0, float(self.link_count) / self.total_link_count)
        else:
            return 0.0

    @property
    def commonness(self) -> float:
        return self.prior_prob

    def __repr__(self):
        return f"<Mention {self.text}->{self.entity.title}>"


class MentionDB:
    def __init__(
        self,
        mention_trie: Trie,
        data_trie: RecordTrie,
        dictionary: Dictionary,
        case_sensitive: bool,
        max_mention_len: int,
        build_params: dict,
        uuid: str,
    ):
        self.mention_trie = mention_trie
        self.data_trie = data_trie
        self.uuid = uuid
        self.build_params = build_params

        self._dictionary = dictionary
        self._case_sensitive = case_sensitive
        self._max_mention_len = max_mention_len

    def __iter__(self) -> Iterator[Mention]:
        for text in self.mention_trie:
            for obj in self.data_trie[text]:
                yield Mention(self._dictionary, text, *obj)

    def query(self, text: str) -> List[Mention]:
        if not self._case_sensitive:
            text = text.lower()

        return [Mention(self._dictionary, text, *o) for o in self.data_trie[text]]

    def prefix_search(self, text: str, start: int = 0) -> List[str]:
        if not self._case_sensitive:
            text = text.lower()

        ret = self.mention_trie.prefixes(text[start : start + self._max_mention_len])
        ret.sort(key=len, reverse=True)
        return ret

    def detect_mentions(self, text: str, tokens: List[Token], entity_indices_in_page: set = set()) -> List[Mention]:
        end_offsets = frozenset([token.end for token in tokens])

        ret = []
        cur = 0

        target_text = text
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
                    candidates = self.data_trie[prefix]
                    if len(candidates) == 1:
                        ret.append(Mention(self._dictionary, text[start:end], *candidates[0], start=start, end=end))
                    else:
                        matched = [c for c in candidates if c[0] in entity_indices_in_page]
                        if len(matched) == 1:
                            (index, *args) = matched[0]
                            ret.append(Mention(self._dictionary, text[start:end], index, *args, start=start, end=end))
                    break

        return ret

    @staticmethod
    def build(
        dump_db: DumpDB,
        dictionary: Dictionary,
        tokenizer: BaseTokenizer,
        min_link_prob: float,
        min_prior_prob: float,
        max_mention_len: int,
        case_sensitive: bool,
        pool_size: int,
        chunk_size: int,
        progressbar: bool = True,
    ) -> "MentionDB":
        start_time = time.time()

        logger.info("Step 1/3: Starting to iterate over Wikipedia pages...")

        name_dict = defaultdict(lambda: Counter())
        init_args = [dump_db, dictionary.serialize(shared_array=True), tokenizer, None]

        with closing(Pool(pool_size, initializer=_init_worker, initargs=init_args)) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_extract_links, max_mention_len=max_mention_len, case_sensitive=case_sensitive)
                for ret in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    for text, index in ret:
                        name_dict[text][index] += 1
                    bar.update(1)

        logger.info("Step 2/3: Starting to count occurrences...")

        name_trie = Trie(name_dict.keys())

        name_counter = Counter()
        init_args[3] = name_trie

        with closing(Pool(pool_size, initializer=_init_worker, initargs=init_args)) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_count_occurrences, max_mention_len=max_mention_len, case_sensitive=case_sensitive)
                for names in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    name_counter.update(names)
                    bar.update(1)

        logger.info("Step 3/3: Building DB...")

        def item_generator():
            for name, entity_counter in name_dict.items():
                doc_count = name_counter[name]
                total_link_count = sum(entity_counter.values())

                if doc_count == 0:
                    continue

                link_prob = float(total_link_count) / doc_count
                if link_prob < min_link_prob:
                    continue

                for index, link_count in entity_counter.items():
                    prior_prob = float(link_count) / total_link_count
                    if prior_prob < min_prior_prob:
                        continue

                    yield (name, (index, link_count, total_link_count, doc_count))

        data_trie = RecordTrie("<IIII", item_generator())
        mention_trie = Trie(data_trie.keys())

        uuid = str(uuid1().hex)

        build_params = dict(
            dump_file=dump_db.dump_file,
            dump_db=dump_db.uuid,
            dictionary=dictionary.uuid,
            build_time=time.time() - start_time,
        )

        return MentionDB(mention_trie, data_trie, dictionary, case_sensitive, max_mention_len, build_params, uuid)

    def serialize(self) -> dict:
        return dict(
            mention_trie=self.mention_trie.tobytes(),
            data_trie=self.data_trie.tobytes(),
            kwargs=dict(
                max_mention_len=self._max_mention_len,
                case_sensitive=self._case_sensitive,
                build_params=self.build_params,
                uuid=self.uuid,
            ),
        )

    def save(self, out_file: str):
        joblib.dump(self.serialize(), out_file)

    @staticmethod
    def load(target: Union[str, dict], dictionary: Dictionary) -> "MentionDB":
        if not isinstance(target, dict):
            target = joblib.load(target)

        if target["kwargs"]["build_params"]["dictionary"] != dictionary.uuid:
            raise RuntimeError("The specified dictionary is different from the one used to build this DB")

        mention_trie = Trie()
        mention_trie = mention_trie.frombytes(target["mention_trie"])
        data_trie = RecordTrie("<IIII")
        data_trie = data_trie.frombytes(target["data_trie"])

        return MentionDB(mention_trie, data_trie, dictionary, **target["kwargs"])


_dictionary: Dictionary
_dump_db: DumpDB
_tokenizer: BaseTokenizer
_name_trie: Trie


def _init_worker(dump_db: DumpDB, dictionary_obj: dict, tokenizer: BaseTokenizer, name_trie: Trie = None):
    global _dump_db, _dictionary, _tokenizer, _name_trie

    _dump_db = dump_db
    _dictionary = Dictionary.load(dictionary_obj)
    _tokenizer = tokenizer
    _name_trie = name_trie


def _extract_links(title: str, max_mention_len: int, case_sensitive: bool) -> List[Tuple[str, int]]:
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


def _count_occurrences(title: str, max_mention_len: int, case_sensitive: bool) -> FrozenSet[str]:
    ret = []

    for paragraph in _dump_db.get_paragraphs(title):
        text = paragraph.text
        tokens = _tokenizer.tokenize(text)

        if not case_sensitive:
            text = text.lower()

        end_offsets = frozenset(token.end for token in tokens)

        for token in tokens:
            start = token.start
            for prefix in _name_trie.prefixes(text[start : start + max_mention_len]):
                if (start + len(prefix)) in end_offsets:
                    ret.append(prefix)

    return frozenset(ret)
