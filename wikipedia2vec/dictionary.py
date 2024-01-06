import logging
import multiprocessing
import time
from collections import Counter
from contextlib import closing
from functools import partial
from itertools import chain
from multiprocessing.pool import Pool
from typing import Any, Iterator, Optional, Union
from uuid import uuid1

import cython
import joblib
import numpy as np
from marisa_trie import Trie, RecordTrie
from tqdm import tqdm

from .dump_db import DumpDB
from .utils.tokenizer.base_tokenizer import BaseTokenizer
from .utils.utils import is_category_title

logger = logging.getLogger(__name__)


@cython.cclass
class Item:
    def __init__(self, index: int, count: int, doc_count: int):
        self.index = index
        self.count = count
        self.doc_count = doc_count


@cython.cclass
class Word(Item):
    def __init__(self, text: str, index: int, count: int, doc_count: int):
        super().__init__(index, count, doc_count)
        self.text = text

    def __reduce__(self):
        return (self.__class__, (self.text, self.index, self.count, self.doc_count))

    def __repr__(self):
        return f"<Word {self.text}>"


@cython.cclass
class Entity(Item):
    def __init__(self, title: str, index: int, count: int, doc_count: int):
        super().__init__(index, count, doc_count)
        self.title = title

    def __reduce__(self):
        return (self.__class__, (self.title, self.index, self.count, self.doc_count))

    def __repr__(self):
        return f"<Entity {self.title}>"


class Dictionary:
    def __init__(
        self,
        word_dict: Trie,
        entity_dict: Trie,
        redirect_dict: RecordTrie,
        word_stats: np.ndarray,
        entity_stats: np.ndarray,
        language: Optional[str] = None,
        lowercase: Optional[bool] = None,
        build_params: Optional[dict] = None,
        min_paragraph_len: int = 0,
        uuid: str = "",
    ):
        self._word_dict = word_dict
        self._entity_dict = entity_dict
        self._redirect_dict = redirect_dict

        # Limit word_stats size in order to handle the case where existing pretrained embedding is larger than it should be
        self._word_stats = word_stats[: len(self._word_dict)]
        self._entity_stats = entity_stats[: len(self._entity_dict)]

        self.min_paragraph_len = min_paragraph_len
        self.uuid = uuid
        self.language = language
        self.lowercase = lowercase
        self.build_params = build_params

        self._entity_offset = len(self._word_dict)

    @property
    def entity_offset(self) -> int:
        return self._entity_offset

    @property
    def word_size(self) -> int:
        return len(self._word_dict)

    @property
    def entity_size(self) -> int:
        return len(self._entity_dict)

    def __len__(self) -> int:
        return len(self._word_dict) + len(self._entity_dict)

    def __iter__(self) -> Iterator[Item]:
        return chain(self.words(), self.entities())

    def words(self) -> Iterator[Word]:
        for word, index in self._word_dict.items():
            yield Word(word, index, *self._word_stats[index].tolist())

    def entities(self) -> Iterator[Entity]:
        for title, index in self._entity_dict.items():
            yield Entity(title, index + self._entity_offset, *self._entity_stats[index].tolist())

    def get_word(self, word: str, default: Any = None) -> Any:
        index = self.get_word_index(word)
        if index == -1:
            return default
        else:
            return Word(word, index, *self._word_stats[index].tolist())

    def get_word_index(self, word: str) -> int:
        try:
            return self._word_dict[word]
        except KeyError:
            return -1

    def get_entity(self, title: str, resolve_redirect: bool = True, default: Any = None) -> Any:
        index = self.get_entity_index(title, resolve_redirect=resolve_redirect)
        if index == -1:
            return default
        else:
            dict_index = index - self._entity_offset
            title = self._entity_dict.restore_key(dict_index)
            return Entity(title, index, *self._entity_stats[dict_index].tolist())

    def get_entity_index(self, title: str, resolve_redirect: bool = True) -> int:
        if resolve_redirect:
            try:
                index = self._redirect_dict[title][0][0]
                return index + self._entity_offset
            except KeyError:
                pass
        try:
            index = self._entity_dict[title]
            return index + self._entity_offset
        except KeyError:
            return -1

    def get_item_by_index(self, index: int) -> Item:
        if index < self._entity_offset:
            return self.get_word_by_index(index)
        else:
            return self.get_entity_by_index(index)

    def get_word_by_index(self, index: int) -> Word:
        word = self._word_dict.restore_key(index)
        return Word(word, index, *self._word_stats[index].tolist())

    def get_entity_by_index(self, index: int) -> Entity:
        dict_index = index - self._entity_offset
        title = self._entity_dict.restore_key(dict_index)
        return Entity(title, index, *self._entity_stats[dict_index].tolist())

    @staticmethod
    def build(
        dump_db: DumpDB,
        tokenizer: BaseTokenizer,
        lowercase: bool,
        min_word_count: int,
        min_entity_count: int,
        min_paragraph_len: int,
        category: bool,
        disambi: bool,
        pool_size: int,
        chunk_size: int,
        progressbar: bool = True,
    ) -> "Dictionary":
        start_time = time.time()

        logger.info("Step 1/2: Processing Wikipedia pages...")

        word_counter = Counter()
        word_doc_counter = Counter()
        entity_counter = Counter()
        entity_doc_counter = Counter()

        with closing(Pool(pool_size, initializer=_init_worker, initargs=(dump_db, tokenizer))) as pool:
            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_process_page, lowercase=lowercase, min_paragraph_len=min_paragraph_len)
                for word_cnt, entity_cnt in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    for word, count in word_cnt.items():
                        word_counter[word] += count
                        word_doc_counter[word] += 1

                    for title, count in entity_cnt.items():
                        if "#" in title:
                            continue
                        entity_counter[title] += count
                        entity_doc_counter[title] += 1

                    bar.update(1)

        logger.info("Step 2/2: Processing Wikipedia redirects...")

        for title, dest_title in dump_db.redirects():
            entity_counter[dest_title] += entity_counter[title]
            del entity_counter[title]

            entity_doc_counter[dest_title] += entity_doc_counter[title]
            del entity_doc_counter[title]

        word_dict = Trie([w for (w, c) in word_counter.items() if c >= min_word_count])
        word_stats = np.zeros((len(word_dict), 2), dtype=np.int32)
        for word, index in word_dict.items():
            word_stats[index][0] = word_counter[word]
            word_stats[index][1] = word_doc_counter[word]

        del word_counter
        del word_doc_counter

        entities = []
        for entity, count in entity_counter.items():
            if count < min_entity_count:
                continue

            if not category and is_category_title(entity, dump_db.language):
                continue

            if not disambi and dump_db.is_disambiguation(entity):
                continue

            entities.append(entity)

        entity_dict = Trie(entities)
        entity_stats = np.zeros((len(entity_dict), 2), dtype=np.int32)
        for entity, index in entity_dict.items():
            entity_stats[index][0] = entity_counter[entity]
            entity_stats[index][1] = entity_doc_counter[entity]

        del entity_counter
        del entity_doc_counter

        redirect_dict = RecordTrie(
            "<I",
            [
                (title, (entity_dict[dest_title],))
                for (title, dest_title) in dump_db.redirects()
                if dest_title in entity_dict
            ],
        )

        build_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            min_word_count=min_word_count,
            min_entity_count=min_entity_count,
            category=category,
            disambi=disambi,
            build_time=time.time() - start_time,
        )

        uuid = str(uuid1().hex)

        logger.info(f"{len(word_dict)} words and {len(entity_dict)} entities are indexed in the dictionary")

        return Dictionary(
            word_dict,
            entity_dict,
            redirect_dict,
            word_stats,
            entity_stats,
            dump_db.language,
            lowercase,
            build_params,
            min_paragraph_len,
            uuid,
        )

    def save(self, out_file: str):
        joblib.dump(self.serialize(), out_file)

    def serialize(self, shared_array: bool = False) -> dict:
        if shared_array:
            word_stats_src = np.asarray(self._word_stats, dtype=np.int32).flatten()
            entity_stats_src = np.asarray(self._entity_stats, dtype=np.int32).flatten()

            word_stats = multiprocessing.RawArray("i", word_stats_src.size)
            entity_stats = multiprocessing.RawArray("i", entity_stats_src.size)
            word_stats[:] = word_stats_src
            entity_stats[:] = entity_stats_src

        else:
            word_stats = np.asarray(self._word_stats, dtype=np.int32)
            entity_stats = np.asarray(self._entity_stats, dtype=np.int32)

        return dict(
            word_dict=self._word_dict.tobytes(),
            entity_dict=self._entity_dict.tobytes(),
            redirect_dict=self._redirect_dict.tobytes(),
            word_stats=word_stats,
            entity_stats=entity_stats,
            meta=dict(
                uuid=self.uuid,
                language=self.language,
                lowercase=self.lowercase,
                min_paragraph_len=self.min_paragraph_len,
                build_params=self.build_params,
            ),
        )

    @staticmethod
    def load(target: Union[str, dict], mmap: bool = True) -> "Dictionary":
        word_dict = Trie()
        entity_dict = Trie()
        redirect_dict = RecordTrie("<I")

        if not isinstance(target, dict):
            if mmap:
                target = joblib.load(target, mmap_mode="r")
            else:
                target = joblib.load(target)

        word_dict.frombytes(target["word_dict"])
        entity_dict.frombytes(target["entity_dict"])
        redirect_dict.frombytes(target["redirect_dict"])

        word_stats = target["word_stats"]
        entity_stats = target["entity_stats"]
        if not isinstance(word_stats, np.ndarray):
            word_stats = np.frombuffer(word_stats, dtype=np.int32).reshape(-1, 2)
            entity_stats = np.frombuffer(entity_stats, dtype=np.int32).reshape(-1, 2)

        return Dictionary(word_dict, entity_dict, redirect_dict, word_stats, entity_stats, **target["meta"])


_dump_db: DumpDB
_tokenizer: BaseTokenizer


def _init_worker(dump_db: DumpDB, tokenizer: BaseTokenizer):
    global _dump_db, _tokenizer

    _dump_db = dump_db
    _tokenizer = tokenizer


def _process_page(title: str, lowercase: bool, min_paragraph_len: int) -> tuple[Counter, Counter]:
    word_counter = Counter()
    entity_counter = Counter()

    for paragraph in _dump_db.get_paragraphs(title):
        entity_counter.update(link.title for link in paragraph.wiki_links)

        tokens = _tokenizer.tokenize(paragraph.text)
        if len(tokens) >= min_paragraph_len:
            if lowercase:
                word_counter.update(token.text.lower() for token in tokens)
            else:
                word_counter.update(token.text for token in tokens)

    return (word_counter, entity_counter)
