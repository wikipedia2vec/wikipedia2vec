#!/bin/bash

cython wikipedia2vec/*.pyx
cython wikipedia2vec/utils/*.pyx
cython wikipedia2vec/utils/tokenizer/*.pyx
cython wikipedia2vec/utils/sentence_detector/*.pyx
