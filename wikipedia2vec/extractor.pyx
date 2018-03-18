# -*- coding: utf-8 -*-
# cython: profile=False

from __future__ import unicode_literals
import logging
import mwparserfromhell
import six

from .utils.wiki_page cimport WikiPage
from .utils.tokenizer import get_tokenizer
from .utils.tokenizer.token cimport Token

logger = logging.getLogger(__name__)


cdef class Paragraph:
    def __init__(self, unicode text, list words, list wiki_links):
        self._text = text
        self._words = words
        self._wiki_links = wiki_links

    property text:
        def __get__(self):
            return self._text

    property words:
        def __get__(self):
            return self._words

    property wiki_links:
        def __get__(self):
            return self._wiki_links

    def __repr__(self):
        if six.PY2:
            return ('<Paragraph %s>' % (' '.join(self._words[:5]) + '...')).encode('utf-8')
        else:
            return '<Paragraph %s>' % (' '.join(self._words[:5]) + '...')

    def __reduce__(self):
        return (self.__class__, (self.text, self._words, self._wiki_links))


cdef class WikiLink:
    def __init__(self, unicode title, unicode text, tuple span):
        self._title = title
        self._text = text
        self._span = span

    property title:
        def __get__(self):
            return self._title

    property text:
        def __get__(self):
            return self._text

    property span:
        def __get__(self):
            return self._span

    def __repr__(self):
        if six.PY2:
            return ('<WikiLink %s->%s>' % (self._text, self._title)).encode('utf-8')
        else:
            return '<WikiLink %s->%s>' % (self._text, self._title)

    def __reduce__(self):
        return (self.__class__, (self._title, self._text, self._span))


cdef class Extractor:
    def __init__(self, unicode language, bint lowercase=True, int min_paragraph_len=20,
                 PrefixSearchable dictionary=None):
        self._language = language
        self._lowercase = lowercase
        self._min_paragraph_len = min_paragraph_len
        self._dictionary = dictionary
        self._tokenizer = get_tokenizer(language)

    cpdef list extract_paragraphs(self, WikiPage page):
        cdef int start, end
        cdef list paragraphs, cur_text, cur_words, cur_links, words
        cdef unicode title, text
        cdef Paragraph p

        paragraphs = []
        cur_text = []
        cur_words = []
        cur_links = []

        if page.is_redirect:
            return []

        for node in self._parse_page(page).nodes:
            if isinstance(node, mwparserfromhell.nodes.Text):
                for (n, paragraph) in enumerate(unicode(node).split('\n')):
                    words = self._extract_words(paragraph)

                    if n == 0:
                        cur_text.append(paragraph)
                        cur_words += words
                    else:
                        paragraphs.append(Paragraph(' '.join(cur_text), cur_words, cur_links))
                        cur_text = [paragraph]
                        cur_words = words
                        cur_links = []

            elif isinstance(node, mwparserfromhell.nodes.Wikilink):
                title = node.title.strip_code()
                if not title:
                    continue

                if node.text:
                    text = node.text.strip_code()
                else:
                    text = node.title.strip_code()

                cur_text.append(text)
                words = self._extract_words(text)
                start = len(cur_words)
                cur_words += words
                end = len(cur_words)
                cur_links.append(WikiLink(self._normalize_title(title), text, (start, end)))

            elif isinstance(node, mwparserfromhell.nodes.Tag):
                if node.tag not in ('b', 'i'):
                    continue
                if not node.contents:
                    continue

                text = node.contents.strip_code()
                cur_text.append(text)
                cur_words += self._extract_words(text)

        return [p for p in paragraphs if (p.words and (p.words[0] not in ('|', '!', '{')) and
                                          len(p.words) >= self._min_paragraph_len)]

    cpdef list _extract_words(self, unicode text):
        cdef int start, end, cur
        cdef bint matched
        cdef frozenset end_offsets
        cdef list words, tokens
        cdef Token token

        tokens = self._tokenizer.tokenize(text)

        if self._dictionary is None:
            if self._lowercase:
                words = [token.text.lower() for token in tokens]
            else:
                words = [token.text for token in tokens]

        else:
            if self._lowercase:
                text = text.lower()

            end_offsets = frozenset([t.span[1] for t in tokens])

            cur = 0
            words = []
            for token in tokens:
                start = token.span[0]
                if cur > start:
                    continue

                matched = False
                for prefix in self._dictionary.prefix_search(text[start:]):
                    end = start + len(prefix)
                    if end in end_offsets:
                        words.append(prefix)
                        cur = end
                        matched = True
                        break

                if not matched:
                    if self._lowercase:
                        words.append(token.text.lower())
                    else:
                        words.append(token.text)

        return words

    cpdef _parse_page(self, WikiPage page):
        try:
            return mwparserfromhell.parse(page.wiki_text)
        except Exception:
            logger.exception('Failed to parse wiki text: %s', page.title)
            return mwparserfromhell.parse('')

    cdef inline unicode _normalize_title(self, unicode title):
        return (title[0].upper() + title[1:]).replace('_', ' ')
