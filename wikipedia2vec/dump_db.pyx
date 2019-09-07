# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import lmdb
import logging
import mwparserfromhell
import pkg_resources
import re
import six
from functools import partial
from uuid import uuid1
import zlib
from contextlib import closing
from six.moves import cPickle as pickle
from multiprocessing.pool import Pool

from .utils.wiki_page cimport WikiPage

logger = logging.getLogger(__name__)

STYLE_RE = re.compile("'''*")


cdef class Paragraph:
    def __init__(self, unicode text, list wiki_links, bint abstract):
        self.text = text
        self.wiki_links = wiki_links
        self.abstract = abstract

    def __repr__(self):
        if six.PY2:
            return ('<Paragraph %s>' % (self.text[:50] + '...')).encode('utf-8')
        else:
            return '<Paragraph %s>' % (self.text[:50] + '...')

    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links, self.abstract))


cdef class WikiLink:
    def __init__(self, unicode title, unicode text, int32_t start, int32_t end):
        self.title = title
        self.text = text
        self.start = start
        self.end = end

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
        if six.PY2:
            return ('<WikiLink %s->%s>' % (self.text, self.title)).encode('utf-8')
        else:
            return '<WikiLink %s->%s>' % (self.text, self.title)

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.start, self.end))


cdef class DumpDB:
    def __init__(self, db_file):
        self._db_file = db_file

        self._env = lmdb.open(db_file, readonly=True, subdir=False, lock=False, max_dbs=3)
        self._meta_db = self._env.open_db(b'__meta__')
        self._page_db = self._env.open_db(b'__page__')
        self._redirect_db = self._env.open_db(b'__redirect__')

    def __reduce__(self):
        return (self.__class__, (self._db_file,))

    @property
    def uuid(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b'id').decode('utf-8')

    @property
    def dump_file(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b'dump_file').decode('utf-8')

    @property
    def language(self):
        with self._env.begin(db=self._meta_db) as txn:
            return txn.get(b'language').decode('utf-8')

    def page_size(self):
        with self._env.begin(db=self._page_db) as txn:
            return txn.stat()['entries']

    def titles(self):
        cdef bytes key

        with self._env.begin(db=self._page_db) as txn:
            cur = txn.cursor()
            for key in cur.iternext(values=False):
                yield key.decode('utf-8')

    def redirects(self):
        cdef bytes key, value

        with self._env.begin(db=self._redirect_db) as txn:
            cur = txn.cursor()
            for (key, value) in iter(cur):
                yield (key.decode('utf-8'), value.decode('utf-8'))

    cpdef unicode resolve_redirect(self, unicode title):
        with self._env.begin(db=self._redirect_db) as txn:
            value = txn.get(title.encode('utf-8'))
            if value:
                return value.decode('utf-8')
            else:
                return title

    cpdef is_redirect(self, unicode title):
        with self._env.begin(db=self._redirect_db) as txn:
            value = txn.get(title.encode('utf-8'))

        return bool(value)

    cpdef is_disambiguation(self, unicode title):
        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(title.encode('utf-8'))

        if not value:
            return False

        return pickle.loads(zlib.decompress(value))[1]

    cpdef list get_paragraphs(self, unicode key):
        cdef bytes value

        with self._env.begin(db=self._page_db) as txn:
            value = txn.get(key.encode('utf-8'))
            if not value:
                raise KeyError(key)

        return self._deserialize_paragraphs(value)

    cdef list _deserialize_paragraphs(self, bytes value):
        cdef list ret, wiki_links

        ret = []
        for obj in pickle.loads(zlib.decompress(value))[0]:
            wiki_links = [WikiLink(*args) for args in obj[1]]
            ret.append(Paragraph(obj[0], wiki_links, obj[2]))

        return ret

    @staticmethod
    def build(dump_reader, out_file, pool_size, chunk_size, preprocess_func=None,
              init_map_size=500000000, buffer_size=3000):
        with closing(lmdb.open(out_file, subdir=False, map_async=True, map_size=init_map_size,
                               max_dbs=3)) as env:
            map_size = [init_map_size]
            meta_db = env.open_db(b'__meta__')
            with env.begin(db=meta_db, write=True) as txn:
                txn.put(b'id', six.text_type(uuid1().hex).encode('utf-8'))
                txn.put(b'dump_file', dump_reader.dump_file.encode('utf-8'))
                txn.put(b'language', dump_reader.language.encode('utf-8'))
                txn.put(b'version', six.text_type(
                    pkg_resources.get_distribution('wikipedia2vec').version).encode('utf-8')
                )

            page_db = env.open_db(b'__page__')
            redirect_db = env.open_db(b'__redirect__')

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
                f = partial(_parse, preprocess_func=preprocess_func)
                for ret in pool.imap_unordered(f, dump_reader, chunksize=chunk_size):
                    if ret:
                        if ret[0] == 'page':
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


def _parse(WikiPage page, preprocess_func):
    cdef int32_t n, start, end
    cdef bint abstract
    cdef unicode title, text, cur_text, wiki_text
    cdef list paragraphs, cur_links, ret

    if page.is_redirect:
        return ('redirect', (page.title.encode('utf-8'), page.redirect.encode('utf-8')))

    # remove style tags to reduce parsing errors
    wiki_text = STYLE_RE.sub('', page.wiki_text)
    try:
        parsed = mwparserfromhell.parse(wiki_text)
    except Exception:
        logger.warn('Failed to parse wiki text: %s', page.title)
        return None

    paragraphs = []
    cur_text = ''
    cur_links = []
    abstract = True

    if preprocess_func is None:
        preprocess_func = lambda x: x

    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.Text):
            for (n, text) in enumerate(six.text_type(node).split('\n')):
                if n == 0:
                    cur_text += preprocess_func(text)
                else:
                    if cur_text and not cur_text.isspace():
                        paragraphs.append([cur_text, cur_links, abstract])

                    cur_text = preprocess_func(text)
                    cur_links = []

        elif isinstance(node, mwparserfromhell.nodes.Wikilink):
            title = node.title.strip_code().strip(' ')
            if title.startswith(':'):
                title = title[1:]
            if not title:
                continue
            title = (title[0].upper() + title[1:]).replace('_', ' ')

            if node.text:
                text = node.text.strip_code()
                # dealing with extended image syntax: https://en.wikipedia.org/wiki/Wikipedia:Extended_image_syntax
                if title.lower().startswith('file:') or title.lower().startswith('image:'):
                    text = text.split('|')[-1]
            else:
                text = node.title.strip_code()

            text = preprocess_func(text)
            start = len(cur_text)
            cur_text += text
            end = len(cur_text)
            cur_links.append((title, text, start, end))

        elif isinstance(node, mwparserfromhell.nodes.ExternalLink):
            if not node.title:
                continue

            text = node.title.strip_code()
            cur_text += preprocess_func(text)

        elif isinstance(node, mwparserfromhell.nodes.Tag):
            if node.tag not in ('b', 'i', 'u'):
                continue
            if not node.contents:
                continue

            text = node.contents.strip_code()
            cur_text += preprocess_func(text)

        elif isinstance(node, mwparserfromhell.nodes.Heading):
            abstract = False

    if cur_text and not cur_text.isspace():
        paragraphs.append([cur_text, cur_links, abstract])

    ret = [paragraphs, page.is_disambiguation]

    return ('page', ((page.title.encode('utf-8'), zlib.compress(pickle.dumps(ret, protocol=-1)))))
