[![Fury badge](https://badge.fury.io/py/wikipedia2vec.png)](http://badge.fury.io/py/wikipedia2vec)

Introduction
------------

Wikipedia2Vec is a tool for obtaining quality vector representations (embeddings) of words and Wikipedia entities from a Wikipedia dump.
This tool is developed and maintained by [Studio Ousia](http://www.ousia.jp).

The main benefits of using Wikipedia2Vec instead of conventional word embedding tools (e.g., Word2Vec, GloVe) are the following:

-   Wikipedia2Vec places words and Wikipedia entities into a same vector space.
    It enables you to easily model the semantics in a text using words and entities in it, which is beneficial for various natural language processing tasks (e.g., text classification, entity linking, and question answering).
-   Conventional word embedding models learn embeddings based only on contextual words of each word in a corpus.
    Wikipedia2Vec uses rich structured information obtained from the internal link structure of Wikipedia entities.

The embeddings are learned efficiently from a Wikipedia dump.
The code is implemented in Python and optimized using Cython, multiprocessing, and BLAS.

This tool is based on the model proposed in this paper: [Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation](https://arxiv.org/abs/1601.01343).

Installation
------------

```
% pip install Wikipedia2Vec
```

Learning Embeddings from a Wikipedia Dump
-----------------------------------------

First, you need to download a Wikipedia dump file (e.g., enwiki-latest-pages-articles.xml.bz2) from [Wikimedia Downloads](https://dumps.wikimedia.org/).

We currently support only English and Japanese, and more languages will be supported in the near future.
Note that, to process Japanese Wikipedia dumps, you need to to install [*mecab-python*](https://pypi.python.org/pypi/mecab-python) package.

Most of the commands explained below have two options *\--pool-size* and *\--chunk-size*, which are used to control [multiprocessing](https://docs.python.org/2/library/multiprocessing.html).

### Building Phrase Dictionary (Optional)

*build\_phrase\_dictionary* constructs a dictionary consisting of phrases extracted from Wikipedia.

```
% wikipedia2vec build_phrase_dictionary DUMP_FILE PHRASE_DIC_NAME
```

The following options can be specified:

-   *\--min-link-count*: The minimum number of occurrences of the target phrase as links in Wikipedia (default: 10)
-   *\--min-link-prob*: The minimum probability that the target phrase appears as a link in Wikipedia (default: 0.1)
-   *\--lowercase/\--no-lowercase*: Whether phrases are lowercased (default: True)
-   *\--max-len*: The maximum number of words in a target phrase (default: 4)

### Building Dictionary

*build\_dictionary* builds a dictionary of words and entities.

```
% wikipedia2vec build_dictionary DUMP_FILE DIC_FILE
```

This command has the following options:

-   *\--phrase*: The phrase dictionary file generated using *build\_phrase\_dictionary* command
-   *\--lowercase/\--no-lowercase*: Whether words are lowercased (default: True)
-   *\--min-word-count*: The minimum number of occurrences of the target word in Wikipedia (default: 10)
-   *\--min-entity-count*: The minimum number of occurrences of the target entity as links in Wikipedia (default: 5)

### Building Link Graph (Optional)

*build\_link\_graph* generates a large sparse matrix representing the link structure of Wikipedia.

```
% wikipedia2vec build_link_graph DUMP_FILE DIC_FILE LINK_GRAPH_FILE
```

There is no specific option in this command.

### Learning Embeddings

```
% wikipedia2vec train_embedding DUMP_FILE DIC_FILE OUT_FILE
```

-   *\--link-graph*: The link graph file generated using *build\_link\_graph*
-   *\--dim-size*: The number of dimensions of the embeddings (default: 300)
-   *\--init-alpha*: The initial learning rate (default: 0.025)
-   *\--min-alpha*: The minimum learning rate (default: 0.0001)
-   *\--window*: The maximum distance between the target item (word or entity) and the predicted word within a text (default: 10)
-   *\--links-per-page*: The number of entities per page used to generate contextual link neighbors (default: 10)
-   *\--negative*: The number of negative samples (default: 5)
-   *\--iteration*: The number of iterations over the Wikipedia (default: 3)
-   *\--sample*: The parameter for downsampling high frequency words (default: 1e-4)

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
