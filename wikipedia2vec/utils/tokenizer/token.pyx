# -*- coding: utf-8 -*-
from __future__ import unicode_literals
import six


cdef class Token:
    def __init__(self, unicode text, tuple span):
        self.text = text
        self.span = span

    def __repr__(self):
        if six.PY2:
            return b'<Token %s>' % self.text.encode('utf-8')
        else:
            return '<Token %s>' % self.text

    def __reduce__(self):
        return (self.__class__, (self.text, self.span))
