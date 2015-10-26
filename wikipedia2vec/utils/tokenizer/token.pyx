# -*- coding: utf-8 -*-


cdef class Token:
    def __init__(self, unicode text, tuple span):
        self.text = text
        self.span = span

    def __repr__(self):
        return '<Token %s>' % self.text.encode('utf-8')

    def __reduce__(self):
        return (self.__class__, (self.text, self.span))
