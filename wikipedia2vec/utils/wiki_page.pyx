# -*- coding: utf-8 -*-

from __future__ import unicode_literals
import re
import six

# obtained from Wikipedia Miner
# https://github.com/studio-ousia/wikipedia-miner/blob/master/src/org/wikipedia/miner/extraction/LanguageConfiguration.java
REDIRECT_REGEXP = re.compile(r"(?:\#|＃)(?:REDIRECT|転送)[:\s]*(?:\[\[(.*)\]\]|(.*))",
                             re.IGNORECASE)


cdef class WikiPage:
    def __init__(self, unicode title, unicode language, unicode wiki_text):
        self._title = title
        self._language = language
        self._wiki_text = wiki_text

    property title:
        def __get__(self):
            return self._title

    property language:
        def __get__(self):
            return self._language

    property wiki_text:
        def __get__(self):
            return self._wiki_text

    def __repr__(self):
        if six.PY2:
            return b'<WikiPage %s>' % self.title.encode('utf-8')
        else:
            return '<WikiPage %s>' % self.title

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

        if dest:
            return self._normalize_title(dest)
        else:
            return None

    cdef inline unicode _normalize_title(self, unicode title):
        return (title[0].upper() + title[1:]).replace('_', ' ')
