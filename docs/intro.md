Introduction
============
---

![Extended Skip-Gram Model](img/model.png)
Wikipedia2Vec is a tool for learning embeddings of words and entities from Wikipedia.
The learned embeddings map similar words and entities close to one another in a continuous vector space.

This tool learns embeddings of words and entities by iterating over entire Wikipedia pages and jointly optimizing the following three submodels:

- *Wikipedia link graph model*, which learns entity embeddings by predicting neighboring entities in Wikipedia's link graph, an undirected graph whose nodes are entities and edges represent links between entities, given each entity in Wikipedia.
Here, an edge is created between a pair of entities if the page of one entity has a link to that of the other entity or if both pages link to each other.

- *Word-based skip-gram model*, which learns word embeddings by predicting neighboring words given each word in a text contained on a Wikipedia page.

- *Anchor context model*, which aims to place similar words and entities near one another in the vector space, and to create interactions between embeddings of words and those of entities.
Here, we obtain referent entities and their neighboring words from links contained in a Wikipedia page, and the model learns embeddings by predicting neighboring words given each entity.

These three submodels are based on the [skip-gram model](https://en.wikipedia.org/wiki/Word2vec#CBOW_and_skip_grams), which is a neural network model with a training objective to find embeddings that are useful for predicting context items (i.e., neighboring words or entities) given a target item.
For further details, please refer to this paper: [Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation](https://arxiv.org/abs/1601.01343).

## Optimized Implementation for Learning Embeddings

Wikipedia2Vec is implemented in Python, and most of its code is converted into C++ using Cython to boost its performance.
Linear algebraic operations required to learn embeddings are performed by Basic Linear Algebra Subprograms (BLAS).
We store the embeddings as a float matrix in a shared memory space and update it in parallel using multiple processes.

## Automatic Generation of Entity Links

One challenge is that many entity names do not appear as links in Wikipedia.
This is because Wikipedia instructs its contributors to [create a link only when the name first occurs on the page](https://en.wikipedia.org/wiki/Wikipedia:Manual_of_Style/Linking#Principles).
This is problematic because Wikipedia2Vec uses links as a source to learn embeddings.

To address this, our tool provides a feature that automatically generates links.
It first creates a dictionary that maps each entity name to its possible referent entities.
This is done by extracting all names and their referring entities from all links contained in Wikipedia.
Then, during training, our tool takes all words and phrases from the target page and converts each into a link to an entity, if the entity is referred by a link on the same page, or if there is only one referent entity associated to the name in the dictionary.