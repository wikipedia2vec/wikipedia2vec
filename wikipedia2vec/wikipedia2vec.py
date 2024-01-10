import ctypes
import logging
import math
import random
import re
import time
from contextlib import closing
from functools import partial
from multiprocessing import RawArray
from multiprocessing.pool import Pool
from typing import Any, Iterable, List, NamedTuple, Optional, Tuple

import cython
import joblib
import numpy as np
from collections import defaultdict
from itertools import islice
from marisa_trie import Trie, RecordTrie
from tqdm import tqdm
from cython.cimports.libcpp.random import mt19937
from cython.cimports.scipy.linalg.cython_blas import saxpy, sdot

from .dictionary import Dictionary
from .dump_db import DumpDB
from .link_graph import LinkGraph
from .mention_db import MentionDB
from .utils.sentence_detector.base_sentence_detector import BaseSentenceDetector
from .utils.tokenizer.base_tokenizer import BaseTokenizer
from cython.cimports.wikipedia2vec.dictionary import Item, Word
from cython.cimports.wikipedia2vec.dump_db import Paragraph, WikiLink
from cython.cimports.wikipedia2vec.mention_db import Mention
from cython.cimports.wikipedia2vec.utils.tokenizer.token import Token
from cython.cimports.wikipedia2vec.utils.sentence_detector.sentence import Sentence

MAX_EXP = cython.declare(cython.float, 6.0)
EXP_TABLE_SIZE = cython.declare(cython.int, 1000)

logger = logging.getLogger(__name__)


class ItemWithScore(NamedTuple):
    item: Item
    score: float


class SharedArrayObject(NamedTuple):
    array: ctypes.Array
    dtype: type
    shape: Tuple[int, ...]


class Wikipedia2Vec:
    def __init__(self, dictionary: Dictionary):
        self._dictionary = dictionary
        self._train_params: Optional[dict] = None

        self.syn0: Optional[np.ndarray] = None
        self.syn1: Optional[np.ndarray] = None

    @property
    def dictionary(self) -> Dictionary:
        return self._dictionary

    @property
    def train_params(self) -> Optional[dict]:
        return self._train_params

    def get_word(self, word: str, default: Any = None) -> Any:
        return self._dictionary.get_word(word, default)

    def get_entity(self, title: str, resolve_redirect: bool = True, default: Any = None) -> Any:
        return self._dictionary.get_entity(title, resolve_redirect, default)

    def get_word_vector(self, word: str) -> np.ndarray:
        if self.syn0 is None:
            raise RuntimeError("The model is not trained yet")

        item = self._dictionary.get_word(word)
        if item is None:
            raise KeyError()

        return self.syn0[item.index]

    def get_entity_vector(self, title: str, resolve_redirect: bool = True) -> np.ndarray:
        if self.syn0 is None:
            raise RuntimeError("The model is not trained yet")

        item = self._dictionary.get_entity(title, resolve_redirect=resolve_redirect)
        if item is None:
            raise KeyError()

        return self.syn0[item.index]

    def get_vector(self, item: Item) -> np.ndarray:
        if self.syn0 is None:
            raise RuntimeError("The model is not trained yet")

        return self.syn0[item.index]

    def most_similar(
        self, item: Item, count: int = 100, min_count: Optional[int] = None
    ) -> List[ItemWithScore[Item, float]]:
        if self.syn0 is None:
            raise RuntimeError("The model is not trained yet")

        vec = self.get_vector(item)
        return self.most_similar_by_vector(vec, count, min_count=min_count)

    def most_similar_by_vector(
        self, vec: np.ndarray, count: int = 100, min_count: Optional[int] = None
    ) -> List[ItemWithScore[Item, float]]:
        if self.syn0 is None:
            raise RuntimeError("The model is not trained yet")

        if min_count is None:
            min_count = 0

        counts = np.concatenate((self._dictionary._word_stats[:, 0], self._dictionary._entity_stats[:, 0]))
        dst = np.dot(self.syn0, vec) / np.linalg.norm(self.syn0, axis=1) / np.linalg.norm(vec)
        dst[counts < min_count] = -100
        indexes = np.argsort(-dst)

        return [
            ItemWithScore(item=self._dictionary.get_item_by_index(int(ind)), score=float(dst[ind]))
            for ind in indexes[:count]
        ]

    def save(self, out_file: str):
        joblib.dump(self.serialize(), out_file)

    def save_text(self, out_file: str, out_format: str = "default"):
        if self.syn0 is None:
            raise RuntimeError("The model is not trained yet")

        with open(out_file, "w") as f:
            if out_format == "word2vec":
                f.write(f"{len(self.dictionary)} {len(self.syn0[0])}\n")

            for item in sorted(self.dictionary, key=lambda o: o.doc_count, reverse=True):
                vec_str = " ".join(f"{v:.3f}" for v in self.get_vector(item))
                if isinstance(item, Word):
                    text = item.text.replace("\t", " ")
                else:
                    text = "ENTITY/" + item.title.replace("\t", " ")

                if out_format in {"word2vec", "glove"}:
                    text = text.replace(" ", "_")
                    f.write(f"{text} {vec_str}\n")
                else:
                    f.write(f"{text}\t{vec_str}\n")

    def serialize(self) -> dict:
        return dict(
            syn0=self.syn0, syn1=self.syn1, dictionary=self._dictionary.serialize(), train_params=self._train_params
        )

    @staticmethod
    def load(in_file: str, numpy_mmap_mode: str = "c"):
        obj = joblib.load(in_file, mmap_mode=numpy_mmap_mode)
        if isinstance(obj["dictionary"], dict):
            dictionary = Dictionary.load(obj["dictionary"])
        else:
            dictionary = obj["dictionary"]  # for backward compatibility

        ret = Wikipedia2Vec(dictionary)
        ret.syn0 = obj["syn0"]
        ret.syn1 = obj["syn1"]
        ret._train_params = obj.get("train_params")

        return ret

    @staticmethod
    def load_text(in_file: str):
        words = defaultdict(int)
        entities = defaultdict(int)
        vectors = []

        with open(in_file, "r") as f:
            if "\t" in list(islice(f, 2))[1]:
                sep = "\t"
            else:
                sep = " "

            f.seek(0)
            n = 0
            for i, line in enumerate(f):
                line = line.rstrip()
                if i == 0 and re.match(r"^\d+\s\d+$", line):  # word2vec format
                    continue

                if sep == "\t":
                    (item_str, vec_str) = line.split(sep)
                    vectors.append(np.array([float(s) for s in vec_str.split(" ")], dtype=np.float32))
                else:
                    items = line.split(sep)
                    item_str = items[0].replace("_", " ")
                    vectors.append(np.array([float(s) for s in items[1:]], dtype=np.float32))

                if item_str.startswith("ENTITY/"):
                    entities[item_str[7:]] = n
                else:
                    words[item_str] = n
                n += 1

        syn0 = np.empty((len(vectors), vectors[0].size))

        word_dict = Trie(words.keys())
        entity_dict = Trie(entities.keys())
        redirect_dict = RecordTrie("<I")

        for word, ind in word_dict.items():
            syn0[ind] = vectors[words[word]]

        entity_offset = len(word_dict)
        for title, ind in entity_dict.items():
            syn0[ind + entity_offset] = vectors[entities[title]]

        word_stats = np.zeros((len(word_dict), 2), dtype=np.int32)
        entity_stats = np.zeros((len(entity_dict), 2), dtype=np.int32)

        dictionary = Dictionary(word_dict, entity_dict, redirect_dict, word_stats, entity_stats)
        ret = Wikipedia2Vec(dictionary)
        ret.syn0 = syn0
        ret.syn1 = None

        return ret

    def train(
        self,
        dump_db: DumpDB,
        link_graph: LinkGraph,
        mention_db: MentionDB,
        tokenizer: BaseTokenizer,
        sentence_detector: Optional[BaseSentenceDetector],
        dim_size: int,
        init_alpha: float,
        min_alpha: float,
        window: int,
        negative: int,
        word_neg_power: float,
        entity_neg_power: float,
        sample: float,
        iteration: int,
        entities_per_page: int,
        pool_size: int,
        chunk_size: int,
        progressbar: bool = True,
    ):
        start_time = time.time()

        logger.info("Building a table for sampling frequent words...")
        word_sampling_table = self._build_word_sampling_table(sample)

        logger.info("Building tables for sampling negatives...")
        word_neg_table = self._build_word_neg_table(word_neg_power)
        entity_neg_table = self._build_entity_neg_table(entity_neg_power)

        total_page_count = dump_db.page_size() * iteration
        if link_graph is not None:
            logger.info("Building a table for iterating links...")
            link_indices = self._build_link_indices(total_page_count, entities_per_page)
        else:
            link_indices = None

        exp_table = self._build_exp_table(max_exp=MAX_EXP, table_size=EXP_TABLE_SIZE)

        logger.info("Initializing weights...")

        vocab_size = len(self.dictionary)

        syn0_arr = (np.random.rand(vocab_size, dim_size).astype(np.float32) - 0.5) / dim_size
        syn1_arr = np.zeros((vocab_size, dim_size), dtype=np.float32)

        init_args = (
            dump_db,
            self.dictionary.serialize(shared_array=True),
            link_graph.serialize(shared_array=True) if link_graph is not None else None,
            mention_db.serialize() if mention_db is not None else None,
            tokenizer,
            sentence_detector,
            _convert_np_array_to_shared_array_object(syn0_arr),
            _convert_np_array_to_shared_array_object(syn1_arr),
            _convert_np_array_to_shared_array_object(word_neg_table),
            _convert_np_array_to_shared_array_object(entity_neg_table),
            _convert_np_array_to_shared_array_object(exp_table),
            _convert_np_array_to_shared_array_object(word_sampling_table),
            _convert_np_array_to_shared_array_object(link_indices) if link_indices is not None else None,
        )

        def args_generator(titles: List[str], iteration: int):
            random.shuffle(titles)
            for n, title in enumerate(titles, len(titles) * iteration):
                alpha = max(min_alpha, init_alpha * (1.0 - float(n) / total_page_count))
                yield (n, title, alpha)

        func = partial(
            _train_page,
            dim_size=dim_size,
            window=window,
            negative=negative,
            entities_per_page=entities_per_page,
        )

        with closing(Pool(pool_size, initializer=_init_worker, initargs=init_args)) as pool:
            titles = list(dump_db.titles())
            for i in range(iteration):
                with tqdm(
                    total=len(titles), mininterval=0.5, disable=not progressbar, desc=f"Iteration {i+1}/{iteration}"
                ) as bar:
                    for _ in pool.imap_unordered(func, args_generator(titles, i), chunksize=chunk_size):
                        bar.update(1)

            logger.info("Terminating pool workers...")

        syn0 = np.frombuffer(syn0_arr, dtype=np.float32).reshape((vocab_size, dim_size))
        syn1 = np.frombuffer(syn1_arr, dtype=np.float32).reshape((vocab_size, dim_size))

        self.syn0 = syn0
        self.syn1 = syn1

        train_params = dict(
            dump_db=dump_db.uuid,
            dump_file=dump_db.dump_file,
            dictionary=self.dictionary.uuid,
            tokenizer=f"{tokenizer.__class__.__module__}.{tokenizer.__class__.__name__}",
            train_time=time.time() - start_time,
            dim_size=dim_size,
            init_alpha=init_alpha,
            min_alpha=min_alpha,
            window=window,
            negative=negative,
            word_neg_power=word_neg_power,
            entity_neg_power=entity_neg_power,
            sample=sample,
            iteration=iteration,
            entities_per_page=entities_per_page,
        )

        if link_graph is not None:
            train_params["link_graph"] = link_graph.uuid

        if mention_db is not None:
            train_params["mention_db"] = mention_db.uuid

        if sentence_detector is not None:
            train_params[
                "sentence_detector"
            ] = f"{sentence_detector.__class__.__module__}.{sentence_detector.__class__.__name__}"

        self._train_params = train_params

    def _build_word_sampling_table(self, sample: float) -> np.ndarray:
        words = list(self.dictionary.words())
        total_word_count = int(sum(w.count for w in words))

        table = np.zeros(len(words), dtype=np.uint32)
        thresh = sample * total_word_count
        for word in words:
            cnt = float(word.count)
            if sample == 0.0:
                word_prob = 1.0
            else:
                word_prob = min(1.0, (np.sqrt(cnt / thresh) + 1) * (thresh / cnt))
            table[word.index] = int(round(word_prob * np.iinfo(np.uint32).max))

        return table

    def _build_word_neg_table(self, power: float) -> np.ndarray:
        if power == 0:
            return self._build_uniform_neg_table(self._dictionary.words())
        else:
            return self._build_unigram_neg_table(self._dictionary.words(), power)

    def _build_entity_neg_table(self, power: float) -> np.ndarray:
        if power == 0:
            return self._build_uniform_neg_table(self._dictionary.entities())
        else:
            return self._build_unigram_neg_table(self._dictionary.entities(), power)

    @staticmethod
    def _build_uniform_neg_table(items: Iterable[Item]) -> np.ndarray:
        return np.array([item.index for item in items], dtype=np.int32)

    @staticmethod
    def _build_unigram_neg_table(items: Iterable[Item], power: float, table_size: int = 100000000) -> np.ndarray:
        items = list(items)
        neg_table = np.zeros(table_size, dtype=np.int32)
        items_pow = float(sum([item.count**power for item in items]))

        index = 0
        cur = items[index].count ** power / items_pow

        for table_index in range(table_size):
            neg_table[table_index] = items[index].index
            if float(table_index) / table_size > cur:
                if index < len(items) - 1:
                    index += 1
                cur += items[index].count ** power / items_pow

        return neg_table

    @staticmethod
    def _build_exp_table(max_exp: float, table_size: int) -> np.ndarray:
        exp_table = np.arange(table_size, dtype=np.float32) / table_size * 2 - 1
        exp_table = np.exp(exp_table * max_exp)
        exp_table = exp_table / (exp_table + 1)

        return exp_table

    def _build_link_indices(self, total_page_count: int, entities_per_page: int) -> np.ndarray:
        offset = self._dictionary.entity_offset
        indices = np.arange(offset, offset + self._dictionary.entity_size, dtype=np.int32)
        rep = int(math.ceil(float(total_page_count) * entities_per_page / indices.size))
        return np.concatenate([np.random.permutation(indices) for _ in range(rep)])


def _convert_np_array_to_shared_array_object(arr: np.ndarray) -> SharedArrayObject:
    return SharedArrayObject(
        array=RawArray(np.ctypeslib.as_ctypes_type(arr.dtype), arr.flatten()),
        dtype=arr.dtype,
        shape=arr.shape,
    )


def _convert_shared_array_object_to_np_array(array_obj: SharedArrayObject) -> np.ndarray:
    return np.frombuffer(array_obj.array, dtype=array_obj.dtype).reshape(*array_obj.shape)


_dump_db: DumpDB
_dictionary: Dictionary
_link_graph: Optional[LinkGraph]
_mention_db: Optional[MentionDB]
_tokenizer: BaseTokenizer
_sentence_detector: Optional[BaseSentenceDetector]
_syn0 = cython.declare(cython.float[:, :])
_syn1 = cython.declare(cython.float[:, :])
_word_neg_table = cython.declare(cython.int[:])
_entity_neg_table = cython.declare(cython.int[:])
_exp_table = cython.declare(cython.float[:])
_word_sampling_table = cython.declare(cython.uint[:])
_link_indices = cython.declare(cython.int[:])
_work = cython.declare(cython.float[:])
_rng = cython.declare(mt19937)


def _init_worker(
    dump_db: DumpDB,
    dictionary_obj: dict,
    link_graph_obj: Optional[dict],
    mention_db_obj: Optional[dict],
    tokenizer: BaseTokenizer,
    sentence_detector: Optional[BaseSentenceDetector],
    syn0_obj: SharedArrayObject,
    syn1_obj: SharedArrayObject,
    word_neg_table: SharedArrayObject,
    entity_neg_table: SharedArrayObject,
    exp_table: SharedArrayObject,
    word_sampling_table: SharedArrayObject,
    link_indices: Optional[SharedArrayObject],
):
    global _dump_db, _dictionary, _link_graph, _mention_db, _tokenizer, _sentence_detector, _syn0, _syn1
    global _word_neg_table, _entity_neg_table, _exp_table, _word_sampling_table, _link_indices, _work, _rng

    _dump_db = dump_db
    _tokenizer = tokenizer
    _word_neg_table = _convert_shared_array_object_to_np_array(word_neg_table)
    _entity_neg_table = _convert_shared_array_object_to_np_array(entity_neg_table)
    _exp_table = _convert_shared_array_object_to_np_array(exp_table)
    _word_sampling_table = _convert_shared_array_object_to_np_array(word_sampling_table)
    if link_indices is None:
        _link_indices = None
    else:
        _link_indices = _convert_shared_array_object_to_np_array(link_indices)

    _dictionary = Dictionary.load(dictionary_obj)

    _syn0 = _convert_shared_array_object_to_np_array(syn0_obj)
    _syn1 = _convert_shared_array_object_to_np_array(syn1_obj)
    _work = np.zeros(_syn0.shape[1], dtype=np.float32)

    np.random.seed()
    _rng = mt19937(np.random.randint(2**31))

    if link_graph_obj is None:
        _link_graph = None
    else:
        _link_graph = LinkGraph.load(link_graph_obj, _dictionary)

    if mention_db_obj is None:
        _mention_db = None
    else:
        _mention_db = MentionDB.load(mention_db_obj, _dictionary)

    if sentence_detector:
        _sentence_detector = sentence_detector
    else:
        _sentence_detector = None


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.locals(
    i=cython.int,
    j=cython.int,
    n=cython.int,
    start=cython.int,
    end=cython.int,
    span_start=cython.int,
    span_end=cython.int,
    entity_start=cython.int,
    entity_end=cython.int,
    word_index=cython.int,
    word_index2=cython.int,
    entity_index=cython.int,
    text_len=cython.int,
    token_len=cython.int,
    alpha=cython.float,
    text=str,
    title=str,
    tokens=list,
    paragraphs=list,
    target_links=list,
    entity_indices_in_page=set,
    token=Token,
    sentence=Sentence,
    paragraph=Paragraph,
    wiki_link=WikiLink,
    mention=Mention,
    word_indices=cython.int[:],
    word_char_positions=cython.int[:],
    sent_char_positions=cython.int[:],
    sent_token_positions=cython.int[:],
    neighbor_entity_indices=cython.int[:],
)
def _train_page(
    arg: Tuple[int, str, float],
    dim_size: cython.int,
    window: cython.int,
    negative: cython.int,
    entities_per_page: cython.int,
):
    n, title, alpha = arg

    # train using Wikipedia link graph
    if _link_graph is not None:
        start = n * entities_per_page
        for i in range(start, start + entities_per_page):
            entity_index = _link_indices[i]
            neighbor_entity_indices = _link_graph.neighbor_indices(entity_index)
            for j in range(len(neighbor_entity_indices)):
                _train_pair(entity_index, neighbor_entity_indices[j], alpha, dim_size, negative, _entity_neg_table)

    paragraphs = _dump_db.get_paragraphs(title)
    if _mention_db is not None:
        entity_indices_in_page = set()
        entity_indices_in_page.add(_dictionary.get_entity_index(title))

        for paragraph in paragraphs:
            for wiki_link in paragraph.wiki_links:
                entity_indices_in_page.add(_dictionary.get_entity_index(wiki_link.title))

        entity_indices_in_page.discard(-1)

    # train using Wikipedia words and anchors
    for paragraph in paragraphs:
        text = paragraph.text
        text_len = len(text)
        tokens = _tokenizer.tokenize(text)
        token_len = len(tokens)
        if not tokens or token_len < _dictionary.min_paragraph_len:
            continue

        word_indices = cython.view.array(shape=(token_len,), itemsize=cython.sizeof(cython.int), format="i")
        word_char_positions = cython.view.array(shape=(text_len + 1,), itemsize=cython.sizeof(cython.int), format="i")

        if _sentence_detector is not None:
            sent_char_positions = cython.view.array(
                shape=(text_len + 1,), itemsize=cython.sizeof(cython.int), format="i"
            )
            sent_token_positions = cython.view.array(shape=(token_len,), itemsize=cython.sizeof(cython.int), format="i")
            sent_char_positions[:] = 0

            for i, sentence in enumerate(_sentence_detector.detect_sentences(text)):
                sent_char_positions[sentence.start : sentence.end] = i

        j = 0
        for i, token in enumerate(tokens):
            if _dictionary.lowercase:
                word_indices[i] = _dictionary.get_word_index(token.text.lower())
            else:
                word_indices[i] = _dictionary.get_word_index(token.text)

            if _sentence_detector is not None:
                sent_token_positions[i] = sent_char_positions[token.start]

            if i > 0:
                word_char_positions[j : token.start] = i - 1
                j = token.start
        word_char_positions[j:] = token_len - 1

        for i in range(len(word_indices)):
            word_index = word_indices[i]
            if word_index == -1:
                continue

            if _word_sampling_table[word_index] < _rng():
                continue

            window = _rng() % window + 1
            start = max(0, i - window)
            end = min(len(word_indices), i + window + 1)
            for j in range(start, end):
                word_index2 = word_indices[j]

                if word_index2 == -1 or i == j:
                    continue

                if _word_sampling_table[word_index2] < _rng():
                    continue

                if _sentence_detector is not None and sent_token_positions[i] != sent_token_positions[j]:
                    continue

                _train_pair(word_index, word_index2, alpha, dim_size, negative, _word_neg_table)

        link_char_flags = np.zeros(text_len + 1, dtype=np.int32)
        target_links = []

        for wiki_link in paragraph.wiki_links:
            entity_index = _dictionary.get_entity_index(wiki_link.title)
            if entity_index == -1:
                continue

            if not (0 <= wiki_link.start <= text_len and 0 <= wiki_link.end <= text_len):
                logger.warn("Detected invalid span of a Wikipedia link")
                continue

            target_links.append((entity_index, wiki_link.start, wiki_link.end))
            link_char_flags[wiki_link.start : wiki_link.end] = 1

        if _mention_db is not None:
            for mention in _mention_db.detect_mentions(text, tokens, entity_indices_in_page):
                if link_char_flags[mention.start : mention.end].sum() == 0:
                    target_links.append((mention.index, mention.start, mention.end))

        for entity_index, entity_start, entity_end in target_links:
            span_start = word_char_positions[entity_start]
            span_end = word_char_positions[max(0, entity_end - 1)] + 1

            window = _rng() % window + 1
            start = max(0, span_start - window)
            end = min(len(word_indices), span_end + window)
            for j in range(start, end):
                word_index2 = word_indices[j]
                if word_index2 == -1 or span_start <= j < span_end:
                    continue

                if _word_sampling_table[word_index2] < _rng():
                    continue

                if _sentence_detector is not None and sent_char_positions[entity_start] != sent_token_positions[j]:
                    continue

                _train_pair(entity_index, word_index2, alpha, dim_size, negative, _word_neg_table)
                _train_pair(word_index2, entity_index, alpha, dim_size, negative, _entity_neg_table)


@cython.cfunc
@cython.inline
@cython.returns(cython.void)
@cython.nogil
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.initializedcheck(False)
@cython.cdivision(True)
@cython.exceptval(check=False)
@cython.locals(
    d=cython.int,
    f=cython.float,
    f_dot=cython.float,
    g=cython.float,
    index=cython.int,
    label=cython.float,
    neg_index=cython.int,
    neg_table_size=cython.int,
    one=cython.int,
    onef=cython.float,
)
def _train_pair(
    index1: cython.int,
    index2: cython.int,
    alpha: cython.float,
    dim_size: cython.int,
    negative: cython.int,
    neg_table: cython.int[:],
):
    one = 1
    onef = 1.0
    neg_table_size = len(neg_table)

    _work[:] = 0

    for d in range(negative + 1):
        if d == 0:
            index = index2
            label = 1.0
        else:
            neg_index = _rng() % neg_table_size
            index = neg_table[neg_index]
            if index == index2:
                continue
            label = 0.0

        f_dot = cython.cast(
            cython.float,
            (
                sdot(
                    cython.address(dim_size),
                    cython.address(_syn0[index1, 0]),
                    cython.address(one),
                    cython.address(_syn1[index, 0]),
                    cython.address(one),
                )
            ),
        )
        if f_dot >= MAX_EXP or f_dot <= -MAX_EXP:
            continue
        f = _exp_table[cython.cast(cython.int, ((f_dot + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2)))]
        g = (label - f) * alpha

        saxpy(
            cython.address(dim_size),
            cython.address(g),
            cython.address(_syn1[index, 0]),
            cython.address(one),
            cython.address(_work[0]),
            cython.address(one),
        )
        saxpy(
            cython.address(dim_size),
            cython.address(g),
            cython.address(_syn0[index1, 0]),
            cython.address(one),
            cython.address(_syn1[index, 0]),
            cython.address(one),
        )

    saxpy(
        cython.address(dim_size),
        cython.address(onef),
        cython.address(_work[0]),
        cython.address(one),
        cython.address(_syn0[index1, 0]),
        cython.address(one),
    )
