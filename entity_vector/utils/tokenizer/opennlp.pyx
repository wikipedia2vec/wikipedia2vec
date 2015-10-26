# -*- coding: utf-8 -*-

import pkg_resources
from jnius import autoclass, JavaClass, MetaJavaClass, JavaMethod

from entity_vector.utils.tokenizer.token cimport Token

File = autoclass('java.io.File')
SentenceModel = autoclass('opennlp.tools.sentdetect.SentenceModel')
SentenceDetectorME = autoclass('opennlp.tools.sentdetect.SentenceDetectorME')
TokenizerModel = autoclass('opennlp.tools.tokenize.TokenizerModel')
TokenizerME = autoclass('opennlp.tools.tokenize.TokenizerME')


cdef class OpenNLPTokenizer:
    cdef object _tokenizer
    cdef OpenNLPSentenceDetector _sentence_detector

    def __init__(self):
        token_model_file = pkg_resources.resource_filename(
            __name__, 'opennlp/en-token.bin'
        )
        tokenizer_model = TokenizerModel(File(token_model_file))
        self._tokenizer = TokenizerME(tokenizer_model)
        self._sentence_detector = OpenNLPSentenceDetector()

    cpdef list tokenize(self, unicode text):
        tokens = []
        for (s_start, s_end) in self._sentence_detector.sent_pos_detect(text):
            for span_ins in self._tokenizer.tokenizePos(text[s_start:s_end]):
                span = (span_ins.getStart() + s_start,
                        span_ins.getEnd() + s_start)
                word = text[span[0]:span[1]]
                token = Token(word, span)
                tokens.append(token)

        return tokens


cdef class OpenNLPSentenceDetector:
    cdef object _detector

    def __init__(self):
        sentence_model_file = pkg_resources.resource_filename(
            __name__, 'opennlp/en-sent.bin'
        )
        sentence_model = SentenceModel(File(sentence_model_file))
        self._detector = SentenceDetectorME(sentence_model)

    cpdef list sent_pos_detect(self, unicode text):
        return [(span.getStart(), span.getEnd())
                for span in self._detector.sentPosDetect(text)]
