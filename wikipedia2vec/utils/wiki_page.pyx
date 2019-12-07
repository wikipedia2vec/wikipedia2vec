# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import re
import six

DISAMBI_REGEXP = re.compile(r"{{\s*(disambiguation|disambig|disamb|dab|geodis)\s*(\||})", re.IGNORECASE)


cdef class WikiPage:
    def __init__(self, unicode title, unicode language, unicode wiki_text, unicode redirect):
        self.title = title
        self.language = language
        self.wiki_text = wiki_text
        self.redirect = redirect

    def __repr__(self):
        if six.PY2:
            return b'<WikiPage %s>' % self.title.encode('utf-8')
        else:
            return '<WikiPage %s>' % self.title

    def __reduce__(self):
        return (self.__class__, (self.title, self.language, self.wiki_text, self.redirect))

    @property
    def is_redirect(self):
        return bool(self.redirect)

    @property
    def is_disambiguation(self):
        return bool(DISAMBI_REGEXP.search(self.wiki_text))
