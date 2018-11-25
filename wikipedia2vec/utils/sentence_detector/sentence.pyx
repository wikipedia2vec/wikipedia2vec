# -*- coding: utf-8 -*-
# cython: profile=False
# License: Apache License 2.0

from __future__ import unicode_literals
import six


cdef class Sentence:
    def __init__(self, unicode text, int32_t start, int32_t end):
        self.text = text
        self.start = start
        self.end = end

    @property
    def span(self):
        return (self.start, self.end)

    def __repr__(self):
        if six.PY2:
            return b'<Sentence %s>' % self.text.encode('utf-8')
        else:
            return '<Sentence %s>' % self.text

    def __reduce__(self):
        return (self.__class__, (self.text, self.start, self.end))
