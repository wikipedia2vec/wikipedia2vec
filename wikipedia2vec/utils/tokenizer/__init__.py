# -*- coding: utf-8 -*-


def get_tokenizer(language):
    if language == 'ja':
        from .mecab_tokenizer import MeCabTokenizer
        return MeCabTokenizer()
    elif language == 'zh':
        from .jieba_tokenizer import JiebaTokenizer
        return JiebaTokenizer()
    else:
        from .regexp_tokenizer import RegexpTokenizer
        return RegexpTokenizer()
