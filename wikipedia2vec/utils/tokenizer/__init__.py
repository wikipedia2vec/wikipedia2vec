# -*- coding: utf-8 -*-

from .mecab_tokenizer import MeCabTokenizer
from .regexp_tokenizer import RegexpTokenizer


def get_tokenizer(language):
    if language == 'en':
        return RegexpTokenizer()
    elif language == 'ja':
        return MeCabTokenizer()
    else:
        raise NotImplementedError('Unsupported language')
