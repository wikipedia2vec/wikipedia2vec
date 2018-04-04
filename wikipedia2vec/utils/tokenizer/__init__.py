# -*- coding: utf-8 -*-


def get_tokenizer(language, phrase_dict=None):
    if language == 'ja':
        from .mecab_tokenizer import MeCabTokenizer
        return MeCabTokenizer(phrase_dict)
    elif language == 'zh':
        from .jieba_tokenizer import JiebaTokenizer
        return JiebaTokenizer(phrase_dict)
    else:
        from .regexp_tokenizer import RegexpTokenizer
        return RegexpTokenizer(phrase_dict)
