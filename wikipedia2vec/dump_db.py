import logging
import pickle
import re
import zlib
from contextlib import closing
from functools import partial
from multiprocessing.pool import Pool
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple
from uuid import uuid1

import cython
import lmdb
import mwparserfromhell

from .utils.utils import CATEGORY_ALIASES, MEDIA_ALIASES, normalize_title
from .utils.wiki_dump_reader import WikiDumpReader
from .utils.wiki_page import WikiPage

logger = logging.getLogger(__name__)


@cython.cclass
class WikiLink:
    title = cython.declare(str, visibility="readonly")
    text = cython.declare(str, visibility="readonly")
    start = cython.declare(cython.int, visibility="readonly")
    end = cython.declare(cython.int, visibility="readonly")

    def __init__(self, title: str, text: str, start: int, end: int):
        self.title = title
        self.text = text
        self.start = start
        self.end = end

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.start, self.end))

    def __repr__(self):
        return f"<WikiLink {self.text}->{self.title}>"


@cython.cclass
class Paragraph:
    text = cython.declare(str, visibility="readonly")
    wiki_links = cython.declare(List[WikiLink], visibility="readonly")
    abstract = cython.declare(cython.bint, visibility="readonly")

    def __init__(self, text: str, wiki_links: List[WikiLink], abstract: bool):
        self.text = text
        self.wiki_links = wiki_links
        self.abstract = abstract

    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links, self.abstract))

    def __repr__(self):
        return f"<Paragraph {self.text[:50] + '...'}>"


class DumpDB:
    def __init__(self, db_file: str):
        self._db_file = db_file

        self._env = lmdb.open(db_file, readonly=True, subdir=False, lock=False, max_dbs=3)
        self._meta_db = self._env.open_db(b"__meta__")
        self._page_db = self._env.open_db(b"__page__")
        self._redirect_db = self._env.open_db(b"__redirect__")

    def __reduce__(self):
        return (self.__class__, (self._db_file,))

    @property
    def uuid(self) -> str:
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b"id").decode("utf-8")

    @property
    def dump_file(self) -> str:
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b"dump_file").decode("utf-8")

    @property
    def language(self) -> str:
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b"language").decode("utf-8")

    def page_size(self) -> int:
        with self._env.begin(db=self._page_db) as txn:
            return txn.stat()["entries"]

    def titles(self) -> Iterator[str]:
        with self._env.begin(db=self._page_db) as txn:
            cur = txn.cursor()
            for key in cur.iternext(values=False):
                yield key.decode("utf-8")

    def redirects(self) -> Iterator[Tuple[str, str]]:
        with self._env.begin(db=self._redirect_db) as txn:
            cur = txn.cursor()
            for key, value in iter(cur):
                yield (key.decode("utf-8"), value.decode("utf-8"))

    def resolve_redirect(self, title: str, max_steps: int = 10) -> str:
        visited = set([title])
        cur_title = title
        for _ in range(max_steps):
            with self._env.begin(db=self._redirect_db) as txn:
                value = txn.get(cur_title.encode("utf-8"))
                if value:
                    cur_title = value.decode("utf-8")
                    if cur_title in visited:
                        logger.warn(f"Detected redirect loop: {title}")
                        return title
                    visited.add(cur_title)

                else:
                    return cur_title

        logger.warn(f"Max steps ({max_steps}) exceeded when resolving redirect: {title}")
        return cur_title

    def is_redirect(self, title: str) -> bool:
        with self._env.begin(db=self._redirect_db) as txn:
            value = txn.get(title.encode("utf-8"))

        return bool(value)

    def is_disambiguation(self, title: str) -> bool:
        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(title.encode("utf-8"))

        if not value:
            return False

        return pickle.loads(zlib.decompress(value))[1]

    def get_paragraphs(self, key: str) -> List[Paragraph]:
        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(key.encode("utf-8"))
            if not value:
                raise KeyError(key)

        return self._deserialize_paragraphs(value)

    def _deserialize_paragraphs(self, value: bytes) -> List[Paragraph]:
        ret = []
        for obj in pickle.loads(zlib.decompress(value))[0]:
            wiki_links = [WikiLink(*args) for args in obj[1]]
            ret.append(Paragraph(obj[0], wiki_links, obj[2]))

        return ret

    @staticmethod
    def build(
        dump_reader: WikiDumpReader,
        out_file: str,
        pool_size: int,
        chunk_size: int,
        preprocess_func: Optional[Callable[[str], str]] = None,
        init_map_size: int = 500000000,
        buffer_size: int = 3000,
    ):
        with closing(lmdb.open(out_file, subdir=False, map_async=True, map_size=init_map_size, max_dbs=3)) as env:
            map_size = [init_map_size]
            meta_db = env.open_db(b"__meta__")
            with env.begin(db=meta_db, write=True) as txn:
                txn.put(b"id", uuid1().hex.encode("utf-8"))
                txn.put(b"dump_file", dump_reader.dump_file.encode("utf-8"))
                txn.put(b"language", dump_reader.language.encode("utf-8"))

            page_db = env.open_db(b"__page__")
            redirect_db = env.open_db(b"__redirect__")

            def write_db(db, data):
                try:
                    with env.begin(db=db, write=True) as txn:
                        txn.cursor().putmulti(data)

                except lmdb.MapFullError:
                    map_size[0] *= 2
                    env.set_mapsize(map_size[0])

                    write_db(db, data)

            with closing(Pool(pool_size)) as pool:
                page_buf = []
                redirect_buf = []
                f = partial(_parse_page, preprocess_func=preprocess_func)
                for ret in pool.imap_unordered(f, dump_reader, chunksize=chunk_size):
                    if ret:
                        if ret[0] == "page":
                            page_buf.append(ret[1])
                        else:  # redirect
                            redirect_buf.append(ret[1])

                    if len(page_buf) == buffer_size:
                        write_db(page_db, page_buf)
                        page_buf = []

                    if len(redirect_buf) == buffer_size:
                        write_db(redirect_db, redirect_buf)
                        redirect_buf = []

                if page_buf:
                    write_db(page_db, page_buf)

                if redirect_buf:
                    write_db(redirect_db, redirect_buf)


_media_re_cache: Dict[str, re.Pattern] = {}
_category_re_cache: Dict[str, re.Pattern] = {}


def _parse_page(page: WikiPage, preprocess_func: Optional[Callable[[str], str]]):
    if page.is_redirect:
        return ("redirect", (page.title.encode("utf-8"), page.redirect.encode("utf-8")))

    # remove style tags to reduce parsing errors
    wiki_text = re.sub("'''*", "", page.wiki_text)
    try:
        parsed_wiki_text = mwparserfromhell.parse(wiki_text)
    except Exception:
        logger.warn(f"Failed to parse wiki text: {page.title}")
        return None

    media_re = _media_re_cache.get(page.language)
    if media_re is None:
        media_prefixes = "|".join(["File", "Image", "Media"] + MEDIA_ALIASES.get(page.language, []))
        media_re = re.compile(f"^(?:{media_prefixes}):", flags=re.IGNORECASE)
        _media_re_cache[page.language] = media_re

    category_re = _category_re_cache.get(page.language)
    if category_re is None:
        category_prefixes = "|".join(["Category"] + CATEGORY_ALIASES.get(page.language, []))
        category_re = re.compile(f"^(?:{category_prefixes}):", flags=re.IGNORECASE)
        _category_re_cache[page.language] = category_re

    class WikiLinkTuple(NamedTuple):
        text: str
        title: str

    def extract_text(node: mwparserfromhell.nodes.Node) -> str:
        text = node.__strip__(normalize=True, collapse=True, keep_template_params=False)
        if text is None:
            return ""

        text = str(text)
        if preprocess_func is not None:
            text = preprocess_func(text)
        return text

    def parse_node(node: mwparserfromhell.nodes.Node) -> list[str | WikiLinkTuple]:
        if isinstance(node, mwparserfromhell.nodes.Wikilink):
            if media_re.match(str(node.title)):  # ignore the link if it points to a media file
                return []

            text = extract_text(node)
            if category_re.match(str(node.title)):
                text = re.sub(category_re, "", text)  # remove the category prefix

            title = node.title.strip_code().strip(" ")
            if title.startswith(":"):
                title = title[1:]
            if not title:
                return [text]

            title = normalize_title(title)
            return [WikiLinkTuple(text=text, title=title)]

        elif isinstance(node, mwparserfromhell.nodes.Tag):
            # ignore references and tables
            if str(node.tag) in {"ref", "table"}:
                return []
            elif not mwparserfromhell.definitions.is_visible(node.tag):
                return []

            parsed_child_nodes = []
            for child_node in node.contents.ifilter(recursive=False):
                parsed_child_nodes += parse_node(child_node)
            return parsed_child_nodes

        text = extract_text(node)
        return [text]

    paragraphs = []
    is_abstract = True
    for section in parsed_wiki_text.get_sections(flat=True, include_lead=True, include_headings=True):
        cur_text = ""
        cur_links = []
        for node in section.ifilter(recursive=False):
            for str_or_wiki_link in parse_node(node):
                if isinstance(str_or_wiki_link, WikiLinkTuple):
                    start = len(cur_text)
                    end = start + len(str_or_wiki_link.text)
                    cur_links.append((str_or_wiki_link.title, str_or_wiki_link.text, start, end))
                    cur_text += str_or_wiki_link.text
                else:  # str
                    cur_text += str_or_wiki_link

        if cur_text:
            paragraphs.append((cur_text, cur_links, is_abstract))
            is_abstract = False

    ret = (tuple(paragraphs), page.is_disambiguation)

    return ("page", ((page.title.encode("utf-8"), zlib.compress(pickle.dumps(ret, protocol=-1)))))
