# -*- coding: utf-8 -*-


def get_sentence_detector(name, language=None):
    if name == 'icu':
        from .icu_sentence_detector import ICUSentenceDetector
        return ICUSentenceDetector(language)
    else:
        raise NotImplementedError()
