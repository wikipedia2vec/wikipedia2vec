# -*- coding: utf-8 -*-
# License: Apache License 2.0

from __future__ import unicode_literals

import bz2
import logging
import re
import six
from xml.etree.cElementTree import iterparse

from .wiki_page cimport WikiPage

logger = logging.getLogger(__name__)

DEFAULT_IGNORED_NS = ('wikipedia:', 'file:', 'portal:', 'template:', 'mediawiki:', 'user:',
                      'help:', 'book:', 'draft:', 'module:', 'timedtext:')
NAMESPACE_RE = re.compile(r"^{(.*?)}")


class WikiDumpReader(object):
    def __init__(self, dump_file, ignored_ns=DEFAULT_IGNORED_NS):
        self._dump_file = dump_file
        self._ignored_ns = ignored_ns

        with bz2.BZ2File(self._dump_file) as f:
            self._language = re.search(r'xml:lang="(.*)"', six.text_type(f.readline())).group(1)

    @property
    def dump_file(self):
        return self._dump_file

    @property
    def language(self):
        return self._language

    def __iter__(self):
        with bz2.BZ2File(self._dump_file) as f:
            c = 0
            for (title, wiki_text, redirect) in _extract_pages(f):
                lower_title = title.lower()
                if any([lower_title.startswith(ns) for ns in self._ignored_ns]):
                    continue
                c += 1

                yield WikiPage(title, self._language, wiki_text, redirect)

                if c % 100000 == 0:
                    logger.info('Processed: %d pages', c)


# obtained from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py
def _extract_pages(in_file):
    elems = (elem for (_, elem) in iterparse(in_file, events=(b'end',)))
    elem = next(elems)

    tag = six.text_type(elem.tag)
    namespace = _get_namespace(tag)
    page_tag = '{%s}page' % namespace
    text_path = './{%s}revision/{%s}text' % (namespace, namespace)
    title_path = './{%s}title' % namespace
    redirect_path = './{%s}redirect' % namespace

    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            text = elem.find(text_path).text or ''
            redirect = elem.find(redirect_path)
            if redirect is not None:
                redirect = _normalize_title(_to_unicode(redirect.attrib['title']))

            yield _to_unicode(title), _to_unicode(text), redirect

            elem.clear()


# obtained from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py
cdef unicode _get_namespace(unicode tag):
    match_obj = NAMESPACE_RE.match(tag)
    if match_obj:
        namespace = match_obj.group(1)
        if not namespace.startswith('http://www.mediawiki.org/xml/export-'):
            raise ValueError('%s not recognized as MediaWiki dump namespace' % namespace)
        return namespace
    else:
        return ''


cdef unicode _to_unicode(basestring s):
    if isinstance(s, unicode):
        return s
    return s.decode('utf-8')


cdef unicode _normalize_title(unicode title):
    return (title[0].upper() + title[1:]).replace('_', ' ')
