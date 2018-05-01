# -*- coding: utf-8 -*-


def get_default_tokenizer(language):
    if language == 'ja':
        return get_tokenizer('mecab')
    elif language == 'zh':
        return get_tokenizer('jieba')
    else:
        return get_tokenizer('regexp')


def get_tokenizer(name, language=None):
    if name == 'regexp':
        from .regexp_tokenizer import RegexpTokenizer
        return RegexpTokenizer().tokenize
    elif name == 'icu':
        from .icu_tokenizer import ICUTokenizer
        return ICUTokenizer(language).tokenize
    elif name == 'mecab':
        from .mecab_tokenizer import MeCabTokenizer
        return MeCabTokenizer().tokenize
    elif name == 'jieba':
        from .jieba_tokenizer import JiebaTokenizer
        return JiebaTokenizer().tokenize
    else:
        raise NotImplementedError()
