#!/bin/bash

cython --cplus -3 wikipedia2vec/dictionary.py
cython --cplus -3 wikipedia2vec/dump_db.py
cython --cplus -3 wikipedia2vec/link_graph.py
cython --cplus -3 wikipedia2vec/mention_db.py
cython --cplus -3 wikipedia2vec/wikipedia2vec.py
cython --cplus -3 wikipedia2vec/utils/sentence_detector/sentence.py
cython --cplus -3 wikipedia2vec/utils/tokenizer/token.py
cython --cplus -3 wikipedia2vec/utils/wiki_page.py
