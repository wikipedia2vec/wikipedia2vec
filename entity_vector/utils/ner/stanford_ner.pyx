# -*- coding: utf-8 -*-

import logging
import pkg_resources
import unicodedata
from jnius import autoclass

logger = logging.Logger(__name__)

_model_path = pkg_resources.resource_filename(
    'entity_vector.utils.ner',
    '/stanford_ner/english.all.3class.distsim.crf.ser.gz'
)

AnswerAnnotation = autoclass('edu.stanford.nlp.ling.CoreAnnotations$AnswerAnnotation')
ArrayList = autoclass('java.util.ArrayList')
Word = autoclass('edu.stanford.nlp.ling.Word')
CRFClassifier = autoclass('edu.stanford.nlp.ie.crf.CRFClassifier')


cdef class EntityMention:
    cdef public tuple span
    cdef public str entity_type

    def __init__(self, tuple span, str entity_type):
        self.span = span
        self.entity_type = entity_type

    def __repr__(self):
        return '<EntityMention %s>' % self.entity_type.encode('utf-8')


cdef class StanfordNER:
    cdef _classifier

    def __init__(self):
        self._classifier = CRFClassifier.getClassifier(_model_path)

    cpdef list extract(self, list words):
        cdef unicode word_str
        cdef str prev_annotation, annotation
        cdef list mentions

        arr = ArrayList()
        for word_str in words:
            arr.add(Word(word_str))

        ret = self._classifier.classifySentence(arr)
        prev_annotation = 'O'
        begin = None
        mentions = []
        for i in range(ret.size()):
            annotation = ret.get(i).get(AnswerAnnotation)
            if prev_annotation != 'O':
                if annotation == 'O' or annotation != prev_annotation:
                    mentions.append(
                        EntityMention((begin, i), prev_annotation)
                    )

            if annotation != 'O' and annotation != prev_annotation:
                begin = i

            prev_annotation = annotation

        if prev_annotation != 'O':
            mentions.append(
                EntityMention((begin, i + 1), prev_annotation)
            )

        return mentions
