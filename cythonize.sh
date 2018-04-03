#!/bin/bash

cython --cplus wikipedia2vec/*.pyx
cython --cplus wikipedia2vec/utils/*.pyx
cython --cplus wikipedia2vec/utils/tokenizer/*.pyx
