Benchmarks
===========
---

We provide benchmark accuracies for the Wikipedia2Vec pre-trained model.  

Evaluations are conducted end-to-end, and you could run evaluation on your learned embeding by running [intrinsic_eval.py](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/scripts/intrinsic_eval.py)

### About The Evaluations
We conducted evaluations on a variety of intrinsic tasks.

Wikipedia2Vec learns embeddings that map words and entities into a unified continuous vector space.  
Thus, we evaluate the learned embeddings with Word Similarity and Word Analogy for words, while we evaluate them with Entity Relatedness for entities.

#### Word Similarity
Word Similarity is a task for intrinsic evaluation of word vectors, which correlates the distance between vectors and human judgments of semantic similarity.

- [MEN-TR-3k](http://clic.cimec.unitn.it/~elia.bruni/MEN.html) ([Bruni et al.,2014](https://staff.fnwi.uva.nl/e.bruni/publications/bruni2014multimodal.pdf))
- [RG-65](https://aclweb.org/aclwiki/RG-65_Test_Collection_(State_of_the_art))
([Rubenstein et al., 1965](https://dl.acm.org/citation.cfm?id=365657))
- [SimLex999](https://www.cl.cam.ac.uk/~fh295/simlex.html) ([Hill et al, 2014](https://arxiv.org/abs/1408.3456?context=cs))
- [WS-353-REL](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) ([Finkelstein et al., 2002](https://dl.acm.org/citation.cfm?id=503110))
- [WS-353-SIM](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) ([Finkelstein et al., 2002](https://dl.acm.org/citation.cfm?id=503110))


#### Word Analogy
Word Analogy is the task, which inspects syntactic,
morphosyntactic and semantic properties of words and phrases.

- [GOOGLE ANALOGY (Syntactic)](http://download.tensorflow.org/data/questions-words.txt) ([Mikolov et al., 2013](https://arxiv.org/pdf/1301.3781))
- [GOOGLE ANALOGY (Semantic)](http://download.tensorflow.org/data/questions-words.txt) ([Mikolov et al., 2013](https://arxiv.org/pdf/1301.3781))

#### Entity Relatedness
Entity Relatedness is the intrinsic evaluation task for entities, where the relatedness between Named Entities are measured.

-  [KORE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/) ([Hoffart et al., 2012](https://dl.acm.org/citation.cfm?id=2396832))


## Model Comparison with Gensim
In this section, we compare the performance on word similarity and analogy
benchmarks of our Wikipedia2Vec pretrained models
with the models trained by [gensim](https://radimrehurek.com/gensim/).

The results on a variety of benckmarks show that Wikipedia2Vec pretrained model
([enwiki_20180420_win10](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_300d.pkl.bz2), 300d) outperforms gensim pretrained model.
For training, we only English Wikipedia dump.
<!-- - enwiki_20180420_win10_300d.pkl
- gensim_model_300d.pkl -->

### Word Similarity
We evaluated the performance on 6 word similarity (relatedness) benchmarks,
and our Wikipedia2Vec pre-trained model outperforms gensim pretrained in all of
these benchmarks.

| Dataset | Wikipedia2Vec | gensim |
|-----------|------------|------------|
| MEN-TR-3k | **0.7541** | 0.7259 |
| RG-65 | **0.7861** | 0.7536 |
| SimLex999 | **0.3578** | 0.3451 |
| WS-353-REL | **0.6435** | 0.6132 |
| WS-353-SIM | **0.7848** | 0.7742 |
| WS-353-ALL | **0.71** | 0.6795 |

### Word Analogy
| Dataset | Wikipedia2Vec | gensim |
|-----------|------------|------------|
| GOOGLE ANALOGY (Semantic) | **0.789** | 0.7516 |
| GOOGLE ANALOGY (Syntactic) | **0.6529** | 0.5719 |

## Model Comparison with word2vec, GloVe
In this section, we compare the performance of
[word2vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/) pretrained embeddings
and our Wikipedia2Vec word embeddings.  

In the previous section, we compared the models only trained with English
Wikipedia dump.
It is widely known that the quality of the word vectors increases significantly
with amount of the training data.
So we evaluate and compare the performances of publicly available
word embeddings trained with much larger amount of training data.

We use [word2vec google_news pre-trained embedding](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
(100B words, 3M vocab),
GloVe's [glove.42B.300d](http://nlp.stanford.edu/data/glove.42B.300d.zip)
(42B tokens, 1.9M vocab) and [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip)
(840B tokens, 2.2M vocab)pre-trained embedding.

<!-- word2vec_gnews_300d.pkl
glove_42b_300d.pkl
glove_840b_300d.pkl -->

### Word Similarity
| Dataset | Wikipedia2Vec | word2vec_gnews | glove_42b_300d | glove_840b_300d|
|-----------|------------|------------|------------|------------|
| MEN-TR-3k | 0.7541 | oov | 0.7362 | **0.8016** |
| RG-65 | 0.7861 | 0.7608 | **0.8171** | 0.7696 |
| SimLex999 | 0.3578 | **0.442** | 0.3738 | 0.4083 |
| WS-353-REL | 0.6435 | 0.6355 | 0.5706 |  **0.6876**  |
| WS-353-SIM | 0.7848 | 0.7717 | 0.6979 | **0.8031** |
| WS-353-ALL | 0.71 | 0.7 |0.6321 | **0.7379** |

### Word Analogy
| Dataset | Wikipedia2Vec | word2vec_gnews | glove_42b_300d | glove_42b_300d|
|-----------|------------|------------|------------|------------|
| GOOG-ANALOGY (Semantic) | 0.7541 | 0.7516 | 0.7092 | **0.7686** |
| GOOGLE ANALOGY (Syntactic) | **0.6529** | 0.5719 | 0.6014 | 0.5909 |


## Model Comparison in English
We also provide benchmark accuracies of Wikipedia2Vec pretrained models
with different training settings to show how the performance varies on various hyperparameters.
All of the pre-trained models are available, and you can download them from the [pretrained](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/) page.

### Word Similarity


### Multilingual Evaluation
### Chinese
