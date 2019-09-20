Learning Embeddings
===================
---

First, you need to download a source Wikipedia dump file (e.g., enwiki-latest-pages-articles.xml.bz2) from [Wikimedia Downloads](https://dumps.wikimedia.org/).
The English dump file can be obtained by running the following command.

```text
% wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
```

Note that you do not need to decompress the dump file.

Then, the embeddings can be trained from a Wikipedia dump using the *train* command.

```text
% wikipedia2vec train DUMP_FILE OUT_FILE
```

**Arguments:**

- *DUMP_FILE*: The Wikipedia dump file
- *OUT_FILE*: The output file

**Options:**

- *--dim-size*: The number of dimensions of the embeddings (default: 100)
- *--window*: The maximum distance between the target item (word or entity) and the context word to be predicted (default: 5)
- *--iteration*: The number of iterations for Wikipedia pages (default: 5)
- *--negative*: The number of negative samples (default: 5)
- *--lowercase/--no-lowercase*: Whether to lowercase words (default: True)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible choices are *regexp*, *icu*, *mecab*, and *jieba*
- *--sent-detect*: The sentence detector used to split texts into sentences. Currently, only *icu* is the possible value (default: None)
- *--min-word-count*: A word is ignored if the total frequency of the word is less than this value (default: 10)
- *--min-entity-count*: An entity is ignored if the total frequency of the entity appearing as the referent of an anchor link is less than this value (default: 5)
- *--min-paragraph-len*: A paragraph is ignored if its length is shorter than this value (default: 5)
- *--category/--no-category*: Whether to include Wikipedia categories in the dictionary (default:False)
- *--disambi/--no-disambi*: Whether to include disambiguation entities in the dictionary (default:False)
- *--link-graph/--no-link-graph*: Whether to learn from the Wikipedia link graph (default: True)
- *--entities-per-page*: For processing each page, the specified number of randomly chosen entities are used to predict their neighboring entities in the link graph (default: 10)
- *--link-mentions*: Whether to convert entity names into links (default: True)
- *--min-link-prob*: An entity name is ignored if the probability of the name appearing as a link is less than this value (default: 0.2)
- *--min-prior-prob*: An entity is not registered as a referent of an entity name if the probability of the entity name referring to the entity is less than this value (default: 0.01)
- *--max-mention-len*: The maximum number of characters in an entity name (default: 20)
- *--init-alpha*: The initial learning rate (default: 0.025)
- *--min-alpha*: The minimum learning rate (default: 0.0001)
- *--sample*: The parameter that controls the downsampling of frequent words (default: 1e-4)
- *--word-neg-power*: Negative sampling of words is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0.75)
- *--entity-neg-power*: Negative sampling of entities is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0)
- *--pool-size*: The number of worker processes (default: the number of CPUs)

The *train* command internally calls the five commands described below (namely, *build-dump-db*, *build-dictionary*, *build-link-graph*, *build-mention-db*, and *train-embedding*).
Further, the learned model file can be converted to a text file compatible with the format of [Word2vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/) using the <a href="#saving-embeddings-in-text-format">save-text</a> command.

## Building Dump Database

The *build-dump-db* command creates a database that contains Wikipedia pages each of which consists of texts and anchor links in it.

```text
% wikipedia2vec build-dump-db DUMP_FILE OUT_FILE
```

**Arguments:**

- *DUMP_FILE*: The Wikipedia dump file
- *OUT_FILE*: The output file

**Options:**

- *--pool-size*: The number of worker processes (default: the number of CPUs)

## Building Dictionary

The *build-dictionary* command builds a dictionary of words and entities.

```text
% wikipedia2vec build-dictionary DUMP_DB_FILE OUT_FILE
```

**Arguments:**

- *DUMP_DB_FILE*: The database file generated using the *build-dump-db* command
- *OUT_FILE*: The output file

**Options:**

- *--lowercase/--no-lowercase*: Whether to lowercase words (default: True)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible choices are *regexp*, *icu*, *mecab*, and *jieba*
- *--min-word-count*: A word is ignored if the total frequency of the word is less than this value (default: 10)
- *--min-entity-count*: An entity is ignored if the total frequency of the entity appearing as the referent of an anchor link is less than this value (default: 5)
- *--min-paragraph-len*: A paragraph is ignored if its length is shorter than this value (default: 5)
- *--category/--no-category*: Whether to include Wikipedia categories in the dictionary (default:False)
- *--disambi/--no-disambi*: Whether to include disambiguation entities in the dictionary (default:False)
- *--pool-size*: The number of worker processes (default: the number of CPUs)

## Building Link Graph (Optional)

The *build-link-graph* command generates a sparse matrix representing the link structure between Wikipedia entities.

```text
% wikipedia2vec build-link-graph DUMP_DB_FILE DIC_FILE OUT_FILE
```

**Arguments:**

- *DUMP_DB_FILE*: The database file generated using the *build-dump-db* command
- *DIC_FILE*: The dictionary file generated by the *build-dictionary* command
- *OUT_FILE*: The output file

**Options:**

- *--pool-size*: The number of worker processes (default: the number of CPUs)

## Building Mention DB (Optional)

The *build-mention-db* command builds a database that contains the mappings of entity names (mentions) and their possible referent entities.

```text
% wikipedia2vec build-mention-db DUMP_DB_FILE DIC_FILE OUT_FILE
```

**Arguments:**

- *DUMP_DB_FILE*: The database file generated using the *build-dump-db* command
- *DIC_FILE*: The dictionary file generated by the *build-dictionary* command
- *OUT_FILE*: The output file

**Options:**

- *--min-link-prob*: An entity name is ignored if the probability of the name appearing as a link is less than this value (default: 0.2)
- *--min-prior-prob*: An entity is not registered as a referent of an entity name if the probability of the entity name referring to the entity is less than this value (default: 0.01)
- *--max-mention-len*: The maximum number of characters in an entity name (default: 20)
- *--case-sensitive*: Whether to detect entity names in a case sensitive manner (default: False)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible choices are *regexp*, *icu*, *mecab*, and *jieba*
- *--pool-size*: The number of worker processes (default: the number of CPUs)

## Learning Embeddings

The *train-embedding* command runs the training of the embeddings.

```text
% wikipedia2vec train-embedding DUMP_DB_FILE DIC_FILE OUT_FILE
```

**Arguments:**

- *DUMP_DB_FILE*: The database file generated using the *build-dump-db* command
- *DIC_FILE*: The dictionary file generated by the *build-dictionary* command
- *OUT_FILE*: The output file

**Options:**

- *--link-graph*: The link graph file generated using the *build-link-graph* command
- *--mention-db*: The mention DB file generated using the *build-mention-db* command
- *--dim-size*: The number of dimensions of the embeddings (default: 100)
- *--window*: The maximum distance between the target item (word or entity) and the context word to be predicted (default: 5)
- *--iteration*: The number of iterations for Wikipedia pages (default: 5)
- *--negative*: The number of negative samples (default: 5)
- *--tokenizer*: The name of the tokenizer used to tokenize a text into words. Possible values are *regexp*, *icu*, *mecab*, and *jieba*
- *--sent-detect*: The sentence detector used to split texts into sentences. Currently, only *icu* is the possible value (default: None)
- *--entities-per-page*: For processing each page, the specified number of randomly chosen entities are used to predict their neighboring entities in the link graph (default: 10)
- *--init-alpha*: The initial learning rate (default: 0.025)
- *--min-alpha*: The minimum learning rate (default: 0.0001)
- *--sample*: The parameter that controls the downsampling of frequent words (default: 1e-4)
- *--word-neg-power*: Negative sampling of words is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0.75)
- *--entity-neg-power*: Negative sampling of entities is performed based on the probability proportional to the frequency raised to the power specified by this option (default: 0)
- *--pool-size*: The number of worker processes (default: the number of CPUs)

## Saving Embeddings in Text Format

*save-text* outputs a model in a text format.

```text
% wikipedia2vec save-text MODEL_FILE OUT_FILE
```

**Arguments:**

- *MODEL_FILE*: The model file generated by the *train-embedding* command
- *OUT_FILE*: The output file

**Options:**

- *--out-format*: The output format. Possible values are *default*, *word2vec*, and *glove*. If *word2vec* and *glove* are specified, the format adopted by [Word2Vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/) are used, respectively.
