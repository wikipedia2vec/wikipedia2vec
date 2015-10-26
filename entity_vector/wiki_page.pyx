# -*- coding: utf-8 -*-

import logging
import mwparserfromhell
import mwparserfromhell.nodes
import re
from collections import defaultdict, Counter
from dawg import BytesDAWG
from repoze.lru import lru_cache

logger = logging.getLogger(__name__)

# obtained from Wikipedia Miner
# https://github.com/studio-ousia/wikipedia-miner/blob/master/src/org/wikipedia/miner/extraction/LanguageConfiguration.java
REDIRECT_REGEXP = re.compile(
    ur"(?:\#|＃)(?:REDIRECT|転送)[:\s]*(?:\[\[(.*)\]\]|(.*))", re.IGNORECASE
)


cdef class Word:
    def __init__(self, unicode text):
        self.text = text

    def __repr__(self):
        return '<Word %s>' % (self.text.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.text,))


cdef class WikiLink:
    def __init__(self, unicode title, unicode text, list words):
        self.title = title
        self.text = text
        self.words = words

    def __repr__(self):
        return '<WikiLink %s>' % (self.title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.title, self.text, self.words))


cdef class WikiPage:
    def __init__(self, title, language, wiki_text):
        self.title = title
        self.language = language
        self.wiki_text = wiki_text

    def __repr__(self):
        return '<WikiPage %s>' % (self.title.encode('utf-8'))

    def __reduce__(self):
        return (self.__class__, (self.title, self.language, self.wiki_text))

    @property
    def is_redirect(self):
        return bool(self.redirect)

    @property
    def redirect(self):
        red_match_obj = REDIRECT_REGEXP.match(self.wiki_text)
        if not red_match_obj:
            return None

        if red_match_obj.group(1):
            dest = red_match_obj.group(1)
        else:
            dest = red_match_obj.group(2)

        return self._normalize_title(dest)

    def extract_paragraphs(self, min_paragraph_len=20):
        cdef int n, start, end, char_start, char_end
        cdef list paragraphs, tokens, items, words, prefixes
        cdef dict end_index
        cdef unicode text, title

        if self.is_redirect:
            return

        paragraphs = [[]]
        tokenizer = _get_tokenizer(self.language)

        for node in _parse_wiki_text(self.title, self.wiki_text).nodes:
            if isinstance(node, mwparserfromhell.nodes.Text):
                for (n, paragraph) in enumerate(unicode(node).split('\n')):
                    tokens = tokenizer.tokenize(paragraph)
                    items = [Word(token.text) for token in tokens]

                    if n == 0:
                        paragraphs[-1] += items
                    else:
                        paragraphs.append(items)

            elif isinstance(node, mwparserfromhell.nodes.Wikilink):
                title = node.title.strip_code()
                if not title:
                    continue

                if node.text:
                    text = node.text.strip_code()
                else:
                    text = node.title.strip_code()

                words = [Word(token.text) for token in tokenizer.tokenize(text)]
                paragraphs[-1].append(
                    WikiLink(self._normalize_title(title), text, words)
                )

            elif isinstance(node, mwparserfromhell.nodes.Tag):
                if node.tag not in ('b', 'i'):
                    continue
                if not node.contents:
                    continue

                words = [
                    Word(token.text)
                    for token in tokenizer.tokenize(node.contents.strip_code())
                ]
                paragraphs[-1] += words

        for paragraph in paragraphs:
            if (
                paragraph and
                (paragraph[0].text and (paragraph[0].text[0] not in ('|', '!', '{'))) and  # remove wikitables
                (len(paragraph) >= min_paragraph_len)  # remove paragraphs that are too short
            ):
                yield paragraph

    cdef inline unicode _normalize_title(self, unicode title):
        title = title[0].upper() + title[1:]
        return title.replace('_', ' ')


@lru_cache(1)
def _get_tokenizer(language):
    if language == 'en':
        from utils.tokenizer.opennlp import OpenNLPTokenizer
        return OpenNLPTokenizer()
    elif language == 'ja':
        from utils.tokenizer.mecab import MeCabTokenizer
        return MeCabTokenizer()
    else:
        raise NotImplementedError('Unsupported language')


@lru_cache(1)
def _parse_wiki_text(title, wiki_text):
    try:
        return mwparserfromhell.parse(wiki_text)

    except Exception:
        logger.exception('Failed to parse wiki text: %s', title)
        return mwparserfromhell.parse('')
