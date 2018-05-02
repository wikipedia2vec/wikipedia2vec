# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import lmdb
import logging
import mwparserfromhell
import re
import six
from uuid import uuid1
import zlib
from contextlib import closing
from six.moves import cPickle as pickle
from multiprocessing.pool import Pool

from .utils.wiki_page cimport WikiPage

logger = logging.getLogger(__name__)

STYLE_RE = re.compile("'''*")


cdef class Paragraph:
    def __init__(self, unicode text, list wiki_links):
        self.text = text
        self.wiki_links = wiki_links

    def __repr__(self):
        if six.PY2:
            return ('<Paragraph %s>' % (self.text[:50] + '...')).encode('utf-8')
        else:
            return '<Paragraph %s>' % (self.text[:50] + '...')

    def __reduce__(self):
        return (self.__class__, (self.text, self.wiki_links))


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
        for obj in pickle.loads(zlib.decompress(value)):
            wiki_links = [WikiLink(*args) for args in obj['links']]
            ret.append(Paragraph(obj['text'], wiki_links))

        return ret

    @staticmethod
    def build(dump_reader, out_file, pool_size, chunk_size, init_map_size=500000000,
              buffer_size=3000):
        with closing(lmdb.open(out_file, subdir=False, map_async=True, map_size=init_map_size,
                               max_dbs=3)) as env:
            map_size = [init_map_size]
            meta_db = env.open_db(b'__meta__')
            with env.begin(db=meta_db, write=True) as txn:
                txn.put(b'id', six.text_type(uuid1().hex).encode('utf-8'))
                txn.put(b'dump_file', dump_reader.dump_file.encode('utf-8'))
                txn.put(b'language', dump_reader.language.encode('utf-8'))

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
                for ret in pool.imap_unordered(_parse, dump_reader, chunksize=chunk_size):
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


def _parse(WikiPage page):
    cdef int32_t n, start, end
    cdef unicode title, text, cur_text, wiki_text
    cdef list ret, cur_links

    if page.is_redirect:
        return ('redirect', (page.title.encode('utf-8'), page.redirect.encode('utf-8')))

    # remove style tags to reduce parsing errors
    wiki_text = STYLE_RE.sub('', page.wiki_text)
    try:
        parsed = mwparserfromhell.parse(wiki_text)
    except Exception:
        logger.warn('Failed to parse wiki text: %s', page.title)
        return None

    ret = []
    cur_text = ''
    cur_links = []

    for node in parsed.nodes:
        if isinstance(node, mwparserfromhell.nodes.Text):
            for (n, text) in enumerate(six.text_type(node).split('\n')):
                if n == 0:
                    cur_text += text
                else:
                    if cur_text and not cur_text.isspace():
                        ret.append(dict(text=cur_text, links=cur_links))

                    cur_text = text
                    cur_links = []

        elif isinstance(node, mwparserfromhell.nodes.Wikilink):
            title = node.title.strip_code()
            if not title:
                continue

            if node.text:
                text = node.text.strip_code()
                # dealing with extended image syntax: https://en.wikipedia.org/wiki/Wikipedia:Extended_image_syntax
                if title.lower().startswith('file:') or title.lower().startswith('image:'):
                    text = text.split('|')[-1]
            else:
                text = node.title.strip_code()

            start = len(cur_text)
            cur_text += text
            end = len(cur_text)
            cur_links.append((_normalize_title(title), text, start, end))

        elif isinstance(node, mwparserfromhell.nodes.ExternalLink):
            if not node.title:
                continue

            text = node.title.strip_code()
            cur_text += text

        elif isinstance(node, mwparserfromhell.nodes.Tag):
            if node.tag not in ('b', 'i', 'u'):
                continue
            if not node.contents:
                continue

            text = node.contents.strip_code()
            cur_text += text

    if cur_text and not cur_text.isspace():
        ret.append(dict(text=cur_text, links=cur_links))

    return ('page', ((page.title.encode('utf-8'), zlib.compress(pickle.dumps(ret, protocol=-1)))))


cdef unicode _normalize_title(unicode title):
    return (title[0].upper() + title[1:]).replace('_', ' ')
