#!/bin/bash

cython -Wextra wikipedia2vec/*.pyx
cython -Wextra wikipedia2vec/utils/*.pyx
cython -Wextra wikipedia2vec/utils/tokenizer/*.pyx
