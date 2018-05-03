Wikipedia2Vec
=============

[![Fury badge](https://badge.fury.io/py/wikipedia2vec.png)](http://badge.fury.io/py/wikipedia2vec)
[![CircleCI](https://circleci.com/gh/studio-ousia/wikipedia2vec/tree/master.svg?style=svg)](https://circleci.com/gh/studio-ousia/wikipedia2vec/tree/master)

Introduction
------------

Wikipedia2Vec is a tool used for obtaining embeddings (vector representations) of words and entities from Wikipedia.
It is developed and maintained by [Studio Ousia](http://www.ousia.jp).

This tool enables you to learn embeddings that map words and entities into a unified continuous vector space.
The embeddings can be used as word embeddings, entity embeddings, and the unified embeddings of words and entities.
They are used in the state-of-the-art models of various tasks such as [entity linking](https://arxiv.org/abs/1601.01343), [named entity recognition](http://www.aclweb.org/anthology/I17-2017), [entity relatedness](https://arxiv.org/abs/1601.01343), and [question answering](https://arxiv.org/abs/1803.08652).

The embeddings can be easily trained from a publicly available Wikipedia dump.
The code is implemented in Python, and optimized using Cython and BLAS.

How It Works
------------

<img src="http://studio-ousia.github.io/wikipedia2vec/img/model.png" width="600" />

Wikipedia2Vec is based on the [Word2vec's skip-gram model](https://en.wikipedia.org/wiki/Word2vec) that learns to predict neighboring words given each word in corpora.
We extend the skip-gram model by adding the following two submodels:

- *The KB link graph model* that learns to estimate neighboring entities given an entity in the link graph of Wikipedia entities.
- *The anchor context model* that learns to predict neighboring words given an entity by using an anchor link that points to the entity and its neighboring words.

By jointly optimizing the skip-gram model and these two submodels, our model simultaneously learns the embedding of words and entities from Wikipedia.
For further details, please refer to our paper: [Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation](https://arxiv.org/abs/1601.01343).

Pretrained Embeddings
---------------------

(coming soon)

Installation
------------

Wikipedia2Vec can be installed from PyPI:

```
% pip install wikipedia2vec
```

This software requires the 64-bit version of Python.
The command installs the following required Python libraries: [click](http://click.pocoo.org/), [jieba](https://github.com/fxsjy/jieba), [joblib](https://pythonhosted.org/joblib/), [lmdb](https://lmdb.readthedocs.io/), [marisa-trie](http://marisa-trie.readthedocs.io/), [mwparserfromhell](https://mwparserfromhell.readthedocs.io/), [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/), [six](https://pythonhosted.org/six/), and [tqdm](https://github.com/tqdm/tqdm).

If you want to train embeddings on your machine, it is highly recommended to install a BLAS library.
We recommend using [OpenBLAS](https://www.openblas.net/) or [Intel Math Kernel Library](https://software.intel.com/en-us/mkl).
Note that, the BLAS library needs to be recognized properly from NumPy.
This can be confirmed by using `numpy.show_config()` function.

To process Japanese Wikipedia dumps, it is also required to install [MeCab](http://taku910.github.io/mecab/) and [its Python binding](https://pypi.python.org/pypi/mecab-python3).

Learning Embeddings
-------------------

First, you need to download a source Wikipedia dump file (e.g., enwiki-latest-pages-articles.xml.bz2) from [Wikimedia Downloads](https://dumps.wikimedia.org/).
The English dump file can be obtained by running the following command.

```
% wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Note that you do not need to decompress the dump file.

Then, the embeddings can be trained from a Wikipedia dump using the *train* command.

```
% wikipedia2vec train DUMP_FILE OUT_FILE
```

**Arguments:**

- *DUMP_FILE*: The Wikipedia dump file
- *OUT_FILE*: The output file

**Options:**

- *--dim-size*: The number of dimensions of the embeddings (default: 100)
- *--window*: The maximum distance between the target item (word or entity) and the context word to be predicted (default: 5)
- *--iteration*: The number of iterations for Wikipedia pages (default: 3)
- *--negative*: The number of negative samples (default: 5)
- *--lowercase/--no-lowercase*: Whether to lowercase words (default: True)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible choices are *regexp*, *icu*, *mecab*, and *jieba*.
- *--min-word-count*: A word is ignored if the total frequency of the word is less than this value (default: 10)
- *--min-entity-count*: An entity is ignored if the total frequency of the entity appearing as the referent of an anchor link is less than this value (default: 5)
- *--min-paragraph-len*: A paragraph is ignored if its length is shorter than this value (default: 5)
- *--category/--no-category*: Whether to include Wikipedia categories in the dictionary (default:False)
- *--link-graph/--no-link-graph*: Whether to learn from the Wikipedia link graph (default: True)
- *--entities-per-page*: For processing each page, the specified number of randomly chosen entities are used to predict their neighboring entities in the link graph (default: 5)
- *--link-mentions*: Whether to detect entity mentions and convert them into links (default: True)
- *--min-link-prob*: A mention surface is ignored if the probability of the surface appearing as an anchor link is less than this value (default: 0.1)
- *--min-prior-prob*: An entity is not registered as a candidate of a mention surface if the probability of the mention surface referring to the entity is less than this value (default: 0.1)
- *--max-mention-len*: The maximum number of characters in a mention surface (default: 20)
- *--init-alpha*: The initial learning rate (default: 0.025)
- *--min-alpha*: The minimum learning rate (default: 0.0001)
- *--sample*: The parameter that controls the downsampling of frequent words (default: 1e-4)
- *--word-neg-power*: Negative sampling of words is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0.75)
- *--entity-neg-power*: Negative sampling of entities is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0)

The *train* command internally calls the five commands described below (namely, *build_dump_db*, *build_dictionary*, *build_link_graph*, *build_mention_db*, and *train_embedding*).

### Building Dump Database

The *build_dump_db* command creates a database that contains Wikipedia pages each of which consists of texts and anchor links in it.
The size of the database based on an English Wikipedia dump is approximately 15GB.

```
% wikipedia2vec build_dump_db DUMP_FILE OUT_FILE
```

**Arguments:**

- *DUMP_FILE*: The Wikipedia dump file
- *OUT_FILE*: The output file

### Building Dictionary

The *build\_dictionary* command builds a dictionary of words and entities.

```
% wikipedia2vec build_dictionary DUMP_DB_FILE OUT_FILE
```

**Arguments:**

- *DUMP_DB_FILE*: The database file generated using the *build\_dump\_db* command
- *OUT_FILE*: The output file

**Options:**

- *--lowercase/--no-lowercase*: Whether to lowercase words (default: True)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible choices are *regexp*, *icu*, *mecab*, and *jieba*.
- *--min-word-count*: A word is ignored if the total frequency of the word is less than this value (default: 10)
- *--min-entity-count*: An entity is ignored if the total frequency of the entity appearing as the referent of an anchor link is less than this value (default: 5)
- *--min-paragraph-len*: A paragraph is ignored if its length is shorter than this value (default: 5)
- *--category/--no-category*: Whether to include Wikipedia categories in the dictionary (default:False)

### Building Link Graph

The *build\_link\_graph* command generates a sparse matrix representing the link structure between Wikipedia entities.

```
% wikipedia2vec build_link_graph DUMP_DB_FILE DIC_FILE OUT_FILE
```

**Arguments:**

- *DUMP_DB_FILE*: The database file generated using the *build\_dump\_db* command
- *DIC_FILE*: The dictionary file generated by the *build\_dictionary* command
- *OUT_FILE*: The output file

There is no option in this command.

### Building Mention DB

The *build\_mention\_db* command builds a database of the mapping from mentions (entity names) to their possible referent entities.

```
% wikipedia2vec build_mention_db DUMP_DB_FILE DIC_FILE OUT_FILE
```

**Arguments:**:

- *DUMP_DB_FILE*: The database file generated using the *build\_dump\_db* command
- *DIC_FILE*: The dictionary file generated by the *build\_dictionary* command
- *OUT_FILE*: The output file

**Options:**

- *--min-link-prob*: A mention surface is ignored if the probability of the surface appearing as an anchor link is less than this value (default: 0.1)
- *--min-prior-prob*: An entity is not registered as a candidate of a mention surface if the probability of the mention surface referring to the entity is less than this value (default: 0.1)
- *--max-mention-len*: The maximum number of characters in a mention surface (default: 20)
- *--case-sensitive*: Whether to detect mentions in a case sensitive manner (default: False)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible choices are *regexp*, *icu*, *mecab*, and *jieba*.

### Learning Embeddings

The *train_embedding* command runs the training of the embeddings.

```
% wikipedia2vec train_embedding DUMP_DB_FILE DIC_FILE OUT_FILE
```

**Arguments:**

- *DUMP_DB_FILE*: The database file generated using the *build\_dump\_db* command
- *DIC_FILE*: The dictionary file generated by the *build\_dictionary* command
- *OUT_FILE*: The output file

**Options:**

- *--link-graph*: The link graph file generated using the *build\_link\_graph* command
- *--mention-db*: The mention DB file generated using the *build\_mention\_db* command
- *--dim-size*: The number of dimensions of the embeddings (default: 100)
- *--window*: The maximum distance between the target item (word or entity) and the context word to be predicted (default: 5)
- *--iteration*: The number of iterations for Wikipedia pages (default: 3)
- *--negative*: The number of negative samples (default: 5)
- *--word-neg-power*: Negative sampling of words is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0.75)
- *--entity-neg-power*: Negative sampling of entities is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0)
- *--entities-per-page*: For processing each page, the specified number of randomly chosen entities are used to predict their neighboring entities in the link graph (default: 10)
- *--init-alpha*: The initial learning rate (default: 0.025)
- *--min-alpha*: The minimum learning rate (default: 0.0001)
- *--sample*: The parameter that controls the downsampling of frequent words (default: 1e-4)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible choices are *regexp*, *icu*, *mecab*, and *jieba*.

### Saving Embeddings in Text Format

*save\_text* outputs a model in a text format.

```
% wikipedia2vec save_text MODEL_FILE OUT_FILE
```

**Arguments:**

- *MODEL_FILE*: The model file generated by the *train\_embedding* command
- *OUT_FILE*: The output file

**Options:**

- *--out-format*: The output format. Possible values are *default*, *word2vec*, and *glove*. If *word2vec* and *glove* are specified, the format adopted by [Word2Vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/) are used, respectively.

Sample Usage
------------

```python
>>> from wikipedia2vec import Wikipedia2Vec

>>> wiki2vec = Wikipedia2Vec.load(MODEL_FILE)

>>> wiki2vec.get_word_vector(u'the')
memmap([ 0.01617998, -0.03325786, -0.01397999, -0.00150471,  0.03237337,
...
       -0.04226106, -0.19677088, -0.31087297,  0.1071524 , -0.09824426], dtype=float32)

>>> wiki2vec.get_entity_vector(u'Scarlett Johansson')
memmap([-0.19793572,  0.30861306,  0.29620451, -0.01193621,  0.18228433,
...
        0.04986198,  0.24383858, -0.01466644,  0.10835337, -0.0697331 ], dtype=float32)

>>> wiki2vec.most_similar(wiki2vec.get_word(u'yoda'), 5)
[(<Word yoda>, 1.0),
 (<Entity Yoda>, 0.84333622),
 (<Word darth>, 0.73328167),
 (<Word kenobi>, 0.7328127),
 (<Word jedi>, 0.7223742)]

>>> wiki2vec.most_similar(wiki2vec.get_entity(u'Scarlett Johansson'), 5)
[(<Entity Scarlett Johansson>, 1.0),
 (<Entity Natalie Portman>, 0.75090045),
 (<Entity Eva Mendes>, 0.73651594),
 (<Entity Emma Stone>, 0.72868186),
 (<Entity Cameron Diaz>, 0.72390842)]
```

Reference
---------

If you use Wikipedia2Vec in a scientific publication, please cite the following paper:

    @InProceedings{yamada-EtAl:2016:CoNLL,
      author    = {Yamada, Ikuya  and  Shindo, Hiroyuki  and  Takeda, Hideaki  and  Takefuji, Yoshiyasu},
      title     = {Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation},
      booktitle = {Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning},
      month     = {August},
      year      = {2016},
      address   = {Berlin, Germany},
      pages     = {250--259},
      publisher = {Association for Computational Linguistics}
    }

License
-------

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
