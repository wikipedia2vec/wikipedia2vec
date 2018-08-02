Benchmarks
===========
---

We provide the performance on a variety of benchmarks for the Wikipedia2Vec pre-trained model.  

Evaluations are conducted end-to-end, and we used  [intrinsic_eval.py](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/scripts/intrinsic_eval.py) under `scripts` directory of Wikipedia2Vec, to evaluate Wikipedia2Vec performance.

### About The Evaluations
We conducted evaluations on a variety of intrinsic tasks.  

Wikipedia2Vec learns embeddings that map words and entities into a unified continuous vector space, and in this experiment, we evaluated the learned embeddings of words and those of entities separately.    
In particular, we evaluated the word embeddings with **Word Similarity** and **Word Analogy**, while we evaluate the entity embeddings with **Entity Relatedness**.

#### Word Similarity
Word Similarity is a task for intrinsic evaluation of word embeddings,
which correlates the distance between vectors and human judgments of semantic similarity.  

- [MEN-TR-3k](http://clic.cimec.unitn.it/~elia.bruni/MEN.html) ([Bruni et al.,2014](https://staff.fnwi.uva.nl/e.bruni/publications/bruni2014multimodal.pdf))
- [RG-65](https://aclweb.org/aclwiki/RG-65_Test_Collection_(State_of_the_art))
([Rubenstein et al., 1965](https://dl.acm.org/citation.cfm?id=365657))
- [SimLex999](https://www.cl.cam.ac.uk/~fh295/simlex.html) ([Hill et al, 2014](https://arxiv.org/abs/1408.3456?context=cs))
- [WS-353-REL](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) ([Finkelstein et al., 2002](https://dl.acm.org/citation.cfm?id=503110))
- [WS-353-SIM](http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/) ([Finkelstein et al., 2002](https://dl.acm.org/citation.cfm?id=503110))


#### Word Analogy
Word Analogy is the task, which inspects syntactic, morphosyntactic and semantic properties of words and phrases.

- [GOOGLE ANALOGY (Syntactic)](http://download.tensorflow.org/data/questions-words.txt) ([Mikolov et al., 2013](https://arxiv.org/pdf/1301.3781))
- [GOOGLE ANALOGY (Semantic)](http://download.tensorflow.org/data/questions-words.txt) ([Mikolov et al., 2013](https://arxiv.org/pdf/1301.3781))

#### Entity Relatedness
Entity Relatedness is the intrinsic evaluation task for entities, where the quality of entity embeddings is evaluated using the human ratings of relatedness between entities.

-  [KORE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/yago-naga/aida/downloads/) ([Hoffart et al., 2012](https://dl.acm.org/citation.cfm?id=2396832))


## Model Comparison with Gensim
In this section, we compare the performance on Word Similarity and Word Analogy
benchmarks of our Wikipedia2Vec-trained model
with the model trained by [gensim](https://radimrehurek.com/gensim/).  
We do not compare the performance on Entity Relatedness here.

For both embeddings, we set the *window size* to 5, *iteration* to 10, and
*negative sampling count* to 15.  
For training, we only used English Wikipedia dump, without adding any additional large-scale corpora.
We used  [gensim_wikipedia.py](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/scripts/gensim_wikipedia.py) under `script` directory of Wikipedia2vec to train gensim-based word embeddings.

The results on a variety of benckmarks show that Wikipedia2Vec pretrained model
([enwiki_20180420_300d.pkl](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2)) outperformed gensim pretrained model.

<!-- - enwiki_20180420_win10_300d.pkl
- gensim_model_300d.pkl -->

### Word Similarity
We evaluated the performance on 6 Word Similarity benchmarks,
and our Wikipedia2Vec pre-trained model outperformed gensim pretrained in almost all of
the benchmarks.

| Dataset | Wikipedia2Vec | gensim |
|-----------|------------|------------|
| MEN-TR-3k | **0.749** | 0.7315 |
| RG-65 | **0.7837** | 0.7582 |
| SimLex999 | **0.3815** | 0.3471 |
| WS-353-ALL | **0.6952** | 0.6933 |
| WS-353-REL | 0.6233 |  **0.625** |
| WS-353-SIM | 0.7597 | **0.7833** |

### Word Analogy
In both of the Word Analogy tasks, the embeddings trained by Wikipedia2Vec
significantly outperformed the embedding trained by gensim.

| Dataset | Wikipedia2Vec | gensim |
|-----------|------------|------------|
| GOOGLE ANALOGY (Semantic) | **0.7892** | 0.782 |
| GOOGLE ANALOGY (Syntactic) | **0.6812** | 0.5783 |

## Model Comparison with word2vec, GloVe
In this section, we compare the performance of
[word2vec](https://code.google.com/archive/p/word2vec/) and [GloVe](https://nlp.stanford.edu/projects/glove/) pretrained embeddings
and our Wikipedia2Vec word embeddings.  

In the previous section, we compared the models only trained with English
Wikipedia dump.  
It is widely known that the quality of the word embeddings increases significantly
with large amount of the training data.
Therefore, we compare the performances of publicly available
word embeddings trained with much larger amount of training data.

We use [word2vec google_news pre-trained embedding](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
(100B words, 3M vocab),
GloVe's [glove.42B.300d](http://nlp.stanford.edu/data/glove.42B.300d.zip)
(42B tokens, 1.9M vocab) and [glove.840B.300d](http://nlp.stanford.edu/data/glove.840B.300d.zip)
(840B tokens, 2.2M vocab) pre-trained embeddings.

### Word Similarity
The glove.840B.300d outperformed our embeddings trained with Wikipedia on
all of the benchmarks, benefiting from its huge vocabulary size and Common Crawl
based huge training corpus.

On the other hand, our Wikipedia2Vec pretrained vector outperformed word2vec_gnews and
glove_42b_300d on some of the benchmarks,
even though training corpora for these embeddings are three or four orders of
magnitudes larger than the training corpus obtained only from Wikipedia.  


| Dataset | Wikipedia2Vec | word2vec_gnews| glove_42b_300d | glove_840b_300d|
|-----------|------------|------------|------------|------------|
| MEN-TR-3k | 0.749 | OOV |0.7362 | **0.8016** |
| RG-65 | 0.7837 | 0.7608 | **0.8171** | 0.7696 |
| SimLex999 | 0.3815 | **0.442** |0.3738 | 0.4083 |
| WS-353-ALL | 0.6952 | 0.7 |0.6321 | **0.7379** |
| WS-353-REL | 0.6233 | 0.6355 |0.5706 |  **0.6876**  |
| WS-353-SIM | 0.7597 | 0.7717 |0.6979 | **0.8031** |


### Word Analogy
In Word Analogy evaluations, we found the same trend as Word Similarity,
and Wikipedia2Vec embeddings shows competitive performance despite of its smaller scale training corpus.
In both of the tasks, The word2vec_gnews shows out-of-vocabulary (OOV).

| Dataset | Wikipedia2Vec | glove_42b_300d | glove_42b_300d|
|-----------|------------|-------------|------------|
| GOOGLE ANALOGY (Semantic) | 0.7892 | **0.8185** | 0.794 |
| GOOGLE ANALOGY (Syntactic) | 0.6812 | 0.6925 | **0.7567** |

## Comparison with State of The Art Entity Embeddings Method
[Ristoski et.al](http://www.semantic-web-journal.net/system/files/swj1495.pdf) proposed **RDF2Vec**, an approach that learns entity embeddings using word embedding methods (i.e., CBOW and skip-gram) with RDF graphs as inputs.

RDF2Vec model has achieved the state of the art performance on KORE dataset.   
We compare Entity Relatedness performance of both of the models with the same word dimensionality.

As shown in the table below, except for the only one category, Hollywood celebrities,
Wikipedia2Vec achieved higher performance on KORE dataset.

| Category | Wikipedia2Vec | RDF2Vec ([Ristoski et.al](http://www.semantic-web-journal.net/system/files/swj1495.pdf)) |
|-----------|------------|------------|
| IT companies | **0.7934** | 0.743 |
| Hollywood Celebrities | 0.6887 | **0.734** |
| Television Series | **0.6415** | 0.635 |
| Video Games | **0.7261** | 0.669 |
| Chuck Norris | **0.6286** | 0.628 |
| All | **0.7084** | 0.692 |

## The Effects of Parameter Tuning
We also provide benchmark evaluations of Wikipedia2Vec pretrained models,
with different training settings to show how the performance varies on various hyper-parameters.
All of the pre-trained models are available, and you can download them from the
[pretrained embeddings page](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).

### Link Graph
The link graph model that learns to estimate neighboring entities given an entity
 in the link graph of Wikipedia entities.  
We compared the performance of the link graph model with the no link graph model to
see the effectiveness of the link graphs between entities.  
Except for the link graph, we set all of the parameters to the same.
For both embeddings, we set the *window size* to 5, *iteration* to 10, and
*negative sampling count* to 15.

#### Word Similarity
In terms of the performance on Word Similarity task, no link graph model
outperforms the link graph model.

| Dataset | Wikipedia2Vec | Wikipedia2Vec (no link graph) |
|-----------|------------|------------|
| MEN-TR-3k | 0.749 | **0.7467** |
| RG-65 | 0.7837 | **0.7987** |
| SimLex999 | 0.3815 | **0.3867** |
| WS-353-ALL | 0.6952 | **0.7009** |
| WS-353-REL | 0.6233 | **0.6304** |
| WS-353-SIM | 0.7597 | **0.7643** |

#### Word Analogy
The link graph model achieves higher performance on both of the Word Analogy task.

| Dataset | Wikipedia2Vec | Wikipedia2Vec (no link graph) |
|-----------|------------|------------|
| GOOGLE ANALOGY (Semantic) | **0.7892** | 0.7804 |
| GOOGLE ANALOGY (Syntactic) | **0.6812** | 0.6703 |

#### Entity Relatedness
Unsurprisingly, without link graph, the model shows significantly huge drop in its
performance in Entity Relatedness tasks, because of the lacking of the information
about the entity relationship.

| Dataset | Wikipedia2Vec | Wikipedia2Vec (no link graph) |
|-----------|------------|------------|
| KORE | **0.6905** | 0.5892 |

### Window Size
Previous work shows that the window size for word embeddings training does matter.
We conducted evaluation on Wikipedia2Vec models with different window sizes,
to see how important the window size is for the performance on intrinsic evaluation tasks.

We compare the performance of [enwiki_20180420 (300d)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2) (Wikipedia2Vec (window=5)) with the one of
[enwiki_20180420_win10 (300d)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_300d.pkl.bz2) (Wikipedia2Vec (window=10)).

For both embeddings, we set the *iteration* to 10, and
*negative sampling count* to 15.  
For training, we only use English Wikipedia dump, without adding any additional large-scale corpora.

#### Word Similarity
Our experimental results show that larger window size seems to improve the
performance on Word Similarity, on the all benchmarks except SimLex999.

It should be noted that [Chiu et al.](http://aclweb.org/anthology/W/W16/W16-2501.pdf) showed
that the performance on word similarity benchmarks like MEN-TR-3k have negative correlations with downstream tasks, while only SimLex999 has high correlation with extrinsic measures.
They also observed that decreasing window size improves the performance on downstream tasks, while it leads to deterioration of performance on most of the Word Similarity benchmarks.

| Dataset | Wikipedia2Vec (window=5)| Wikipedia2Vec (window=10) |
|-----------|------------|------------|
| MEN-TR-3k | 0.749 | **0.7541** |
| RG-65 | 0.7837 | **0.7861** |
| SimLex999 | **0.3815** | 0.3578 |
| WS-353-ALL | 0.6952 | **0.71** |
| WS-353-REL | 0.6233 | **0.6435** |
| WS-353-SIM | 0.7597 | **0.7848** |

#### Word Analogy

| Dataset | Wikipedia2Vec (window=5)| Wikipedia2Vec (window=10) |
|-----------|------------|------------|
| GOOGLE ANALOGY (Semantic) | **0.7892** | 0.789 |
| GOOGLE ANALOGY (Syntactic) | **0.6812** | 0.6529 |

#### Entity Relatedness
| Dataset | Wikipedia2Vec (window=5)| Wikipedia2Vec (window=10) |
|-----------|------------|------------|
| KORE | **0.6905** | 0.6811 |

## Multilingual Evaluation
We evaluate our word embeddings (300d) on German, French, Spanish, Portuguese, Polish and Chinese
word analogy task.

#### Multilingual Word Analogy Dataset
We use the same word analogy dataset as [Grave et.al](https://arxiv.org/abs/1802.06893).

- [Google Analogy Dataset (German)](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/data/de/word/analogy/DE-GOOGLE.txt)
- [Google Analogy Dataset (French)](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/data/fr/word/analogy/FR-GOOGLE.txt)
- [Google Analogy Dataset (Spanish)](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/data/es/word/analogy/ES-GOOGLE.txt)
- [Google Analogy Dataset (Portuguese)](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/data/pt/word/analogy/PT-GOOGLE.txt)
- [Google Analogy Dataset (Polish)](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/data/pl/word/analogy/PL-GOOGLE.txt)
- [Google Analogy Dataset (Chinese)](https://github.com/wikipedia2vec/wikipedia2vec/blob/master/data/zh/word/analogy/ZH-GOOGLE.txt)

#### Multilingual Word Analogy Results
The results of word analogy task evaluation are as follows.  
Following [Grave et.al](https://arxiv.org/abs/1802.06893), we restrict the vocabulary for the analogy tasks to the 200,000 most frequent words from the training data.

| language | Wikipedia2Vec | fastText ([Grave et.al](https://arxiv.org/abs/1802.06893))|
|-----------|------------|------------|
| German| **0.617** | 0.61 |
| French | **0.68** | 0.642 |
| Spanish | **0.574** | **0.574** |
| Portuguese | 0.53 | **0.54** |
| Polish | 0.516 | **0.534** |
| Chinese| 0.572 | **0.631** |
