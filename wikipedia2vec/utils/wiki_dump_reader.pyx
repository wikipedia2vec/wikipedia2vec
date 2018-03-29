# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import bz2
import logging
import re
import six
from xml.etree.cElementTree import iterparse

from .wiki_page import WikiPage

logger = logging.getLogger(__name__)

DEFAULT_IGNORED_NS = ('wikipedia:', 'file:', 'portal:', 'template:', 'mediawiki:', 'user:',
                      'help:', 'book:', 'draft:')
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
            for (title, wiki_text) in extract_pages(f):
                if any([title.lower().startswith(ns) for ns in self._ignored_ns]):
                    continue
                c += 1

                yield WikiPage(six.text_type(title), self._language, six.text_type(wiki_text))

                if c % 100000 == 0:
                    logger.info('Processed: %d', c)


# obtained from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py
def get_namespace(tag):
    match_obj = NAMESPACE_RE.match(tag)
    if match_obj:
        namespace = match_obj.group(1)
        if not namespace.startswith("http://www.mediawiki.org/xml/export-"):
            raise ValueError("%s not recognized as MediaWiki dump namespace" % namespace)
        return namespace
    else:
        return ''


# obtained from https://github.com/RaRe-Technologies/gensim/blob/develop/gensim/corpora/wikicorpus.py
def extract_pages(in_file):
    elems = (elem for (_, elem) in iterparse(in_file, events=('end',)))
    elem = next(elems)

    namespace = get_namespace(elem.tag)
    page_tag = '{%s}page' % namespace
    text_path = './{%s}revision/{%s}text' % (namespace, namespace)
    title_path = './{%s}title' % namespace

    for elem in elems:
        if elem.tag == page_tag:
            title = elem.find(title_path).text
            text = elem.find(text_path).text

            yield title, text or ''

            elem.clear()
