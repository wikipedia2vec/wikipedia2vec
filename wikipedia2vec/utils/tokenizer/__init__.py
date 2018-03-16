# -*- coding: utf-8 -*-
from __future__ import absolute_import
from .mecab_tokenizer import MeCabTokenizer
from .regexp_tokenizer import RegexpTokenizer


def get_tokenizer(language):
    if language == 'en':
        return RegexpTokenizer()
    elif language == 'ja':
        return MeCabTokenizer()
    else:
        raise NotImplementedError('Unsupported language')
