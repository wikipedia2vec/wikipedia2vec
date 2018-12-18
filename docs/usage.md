API Usage
===========
---

Wikipedia2Vec provides functions to access the learned embeddings of words and entities.

The embeddings can be loaded by the `load()` method:

```python
>>> from wikipedia2vec import Wikipedia2Vec
>>> wiki2vec = Wikipedia2Vec.load(MODEL_FILE)
```

The embeddings of words and those of entities can be obtained using the `get_word_vector()` and `get_entity_vector()` methods, respectively:

```python
>>> wiki2vec.get_word_vector('the')
memmap([ 0.01617998, -0.03325786, -0.01397999, -0.00150471,  0.03237337,
...
       -0.04226106, -0.19677088, -0.31087297,  0.1071524 , -0.09824426], dtype=float32)

>>> wiki2vec.get_entity_vector('Scarlett Johansson')
memmap([-0.19793572,  0.30861306,  0.29620451, -0.01193621,  0.18228433,
...
        0.04986198,  0.24383858, -0.01466644,  0.10835337, -0.0697331 ], dtype=float32)
```

Furthermore, the `most_similar()` method takes an item (i.e., words or entities), and computes most similar items of the item in the vector space based on cosine similarity.
The number of items to be returned can be specified as a second argument:

```python
>>> wiki2vec.most_similar(wiki2vec.get_word('yoda'), 5)
[(<Word yoda>, 1.0),
 (<Entity Yoda>, 0.84333622),
 (<Word darth>, 0.73328167),
 (<Word kenobi>, 0.7328127),
 (<Word jedi>, 0.7223742)]

>>> wiki2vec.most_similar(wiki2vec.get_entity('Scarlett Johansson'), 5)
[(<Entity Scarlett Johansson>, 1.0),
 (<Entity Natalie Portman>, 0.75090045),
 (<Entity Eva Mendes>, 0.73651594),
 (<Entity Emma Stone>, 0.72868186),
 (<Entity Cameron Diaz>, 0.72390842)]
```