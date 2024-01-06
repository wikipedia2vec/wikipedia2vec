import logging
import multiprocessing
import time
from contextlib import closing
from functools import partial
from itertools import chain
from multiprocessing.pool import Pool
from typing import Optional, List, Tuple, Union
from uuid import uuid1

import joblib
import numpy as np
from scipy.sparse import coo_matrix
from tqdm import tqdm

from .dictionary import Dictionary, Entity
from .dump_db import DumpDB

logger = logging.getLogger(__name__)


class LinkGraph:
    def __init__(
        self, dictionary: Dictionary, indices: np.ndarray, indptr: np.ndarray, build_params: dict, uuid: str = ""
    ):
        self.uuid = uuid
        self.build_params = build_params
        self._dictionary = dictionary
        self._indptr = indptr
        self._indices = indices

        self._offset = self._dictionary.entity_offset

    def neighbors(self, item: Entity) -> List[Entity]:
        return [self._dictionary.get_entity_by_index(i) for i in self.neighbor_indices(item.index).tolist()]

    def neighbor_indices(self, index: int) -> np.ndarray:
        index -= self._offset
        return self._indices[self._indptr[index] : self._indptr[index + 1]]

    def serialize(self, shared_array: bool = False) -> dict:
        if shared_array:
            indices = multiprocessing.RawArray("i", len(self._indices))
            indptr = multiprocessing.RawArray("i", len(self._indptr))

            indices[:] = self._indices
            indptr[:] = self._indptr
        else:
            indices = np.asarray(self._indices, dtype=np.int32)
            indptr = np.asarray(self._indptr, dtype=np.int32)

        return dict(indices=indices, indptr=indptr, build_params=self.build_params, uuid=self.uuid)

    def save(self, out_file: str):
        joblib.dump(self.serialize(), out_file)

    @staticmethod
    def load(target: Union[str, dict], dictionary: Dictionary, mmap: bool = True) -> "LinkGraph":
        if not isinstance(target, dict):
            if mmap:
                target = joblib.load(target, mmap_mode="r")
            else:
                target = joblib.load(target)

        if target["build_params"]["dictionary"] != dictionary.uuid:
            raise RuntimeError("The specified dictionary is different from the one used to build this link graph")

        indices = target.pop("indices")
        indptr = target.pop("indptr")
        if not isinstance(indices, np.ndarray):
            indices = np.frombuffer(indices, dtype=np.int32)
            indptr = np.frombuffer(indptr, dtype=np.int32)

        return LinkGraph(dictionary, indices, indptr, **target)

    @staticmethod
    def build(
        dump_db: DumpDB, dictionary: Dictionary, pool_size: int, chunk_size: int, progressbar: bool = True
    ) -> "LinkGraph":
        start_time = time.time()

        logger.info("Step 1/2: Processing Wikipedia pages...")

        with closing(
            Pool(pool_size, initializer=_init_worker, initargs=(dump_db, dictionary.serialize(shared_array=True)))
        ) as pool:
            rows = []
            cols = []

            with tqdm(total=dump_db.page_size(), mininterval=0.5, disable=not progressbar) as bar:
                f = partial(_process_page, offset=dictionary.entity_offset)

                for ret in pool.imap_unordered(f, dump_db.titles(), chunksize=chunk_size):
                    if ret:
                        page_indices, dest_indices = ret
                        rows.append(page_indices)
                        rows.append(dest_indices)
                        cols.append(dest_indices)
                        cols.append(page_indices)

                    bar.update(1)

        logger.info("Step 2/2: Converting matrix...")

        rows = list(chain(*rows))
        cols = list(chain(*cols))

        matrix = coo_matrix(
            (np.ones(len(rows), dtype=bool), (np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32))),
            dtype=bool,
        )
        del rows, cols

        matrix = matrix.tocsr()

        matrix.indices += dictionary.entity_offset

        build_params = dict(
            dump_file=dump_db.dump_file,
            dump_db=dump_db.uuid,
            dictionary=dictionary.uuid,
            build_time=time.time() - start_time,
        )

        uuid = str(uuid1().hex)

        return LinkGraph(dictionary, matrix.indices, matrix.indptr, build_params, uuid)


_dump_db: DumpDB
_dictionary: Dictionary


def _init_worker(dump_db: DumpDB, dictionary_obj: dict):
    global _dump_db, _dictionary

    _dump_db = dump_db
    _dictionary = Dictionary.load(dictionary_obj)


def _process_page(title: str, offset: int) -> Optional[Tuple[List[int], List[int]]]:
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

    return ([page_index] * len(dest_indices), dest_indices)
