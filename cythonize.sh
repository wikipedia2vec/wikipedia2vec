#!/bin/bash

cythonize -f wikipedia2vec/*.pyx
cythonize -f wikipedia2vec/utils/*.pyx
cythonize -f wikipedia2vec/utils/tokenizer/*.pyx
