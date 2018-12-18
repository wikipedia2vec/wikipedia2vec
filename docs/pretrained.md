# Pretrained Embeddings
---

We provide pretrained embeddings for 12 languages in binary and text format.
The binary files can be loaded using the `Wikipedia2Vec.load()` method (see [API Usage](usage.md)).
The text files are compatible with the text format of [Word2vec](https://code.google.com/archive/p/word2vec/).
Therefore, these files can be loaded using other libraries such as Gensim's [`load_word2vec_format()`](https://radimrehurek.com/gensim/models/keyedvectors.html#gensim.models.keyedvectors.Word2VecKeyedVectors.load_word2vec_format).
In the text files, all entities have a prefix *ENTITY/* to distinguish them from words.

#### English

- *enwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_300d.txt.bz2)
  [500d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_500d.pkl.bz2)
  [500d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_500d.txt.bz2)
- *enwiki_20180420_nolg* (window=5, iteration=10, negative=15, no link graph):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_300d.txt.bz2)
  [500d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_500d.pkl.bz2)
  [500d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_nolg_500d.txt.bz2)
- *enwiki_20180420_win10* (window=10, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_300d.txt.bz2)
  [500d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_500d.pkl.bz2)
  [500d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/en/2018-04-20/enwiki_20180420_win10_500d.txt.bz2)

#### Arabic

- *arwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/ar/2018-04-20/arwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/ar/2018-04-20/arwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/ar/2018-04-20/arwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/ar/2018-04-20/arwiki_20180420_300d.txt.bz2)

#### Chinese

- *zhwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/zh/2018-04-20/zhwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/zh/2018-04-20/zhwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/zh/2018-04-20/zhwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/zh/2018-04-20/zhwiki_20180420_300d.txt.bz2)

#### Dutch

- *nlwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/nl/2018-04-20/nlwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/nl/2018-04-20/nlwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/nl/2018-04-20/nlwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/nl/2018-04-20/nlwiki_20180420_300d.txt.bz2)

#### French

- *frwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/fr/2018-04-20/frwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/fr/2018-04-20/frwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/fr/2018-04-20/frwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/fr/2018-04-20/frwiki_20180420_300d.txt.bz2)

#### German

- *dewiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/de/2018-04-20/dewiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/de/2018-04-20/dewiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/de/2018-04-20/dewiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/de/2018-04-20/dewiki_20180420_300d.txt.bz2)

#### Italian

- *itwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/it/2018-04-20/itwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/it/2018-04-20/itwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/it/2018-04-20/itwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/it/2018-04-20/itwiki_20180420_300d.txt.bz2)

#### Japanese
- *jawiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/ja/2018-04-20/jawiki_20180420_300d.txt.bz2)

#### Polish

- *plwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/pl/2018-04-20/plwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/pl/2018-04-20/plwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/pl/2018-04-20/plwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/pl/2018-04-20/plwiki_20180420_300d.txt.bz2)

#### Portuguese

- *ptwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/pt/2018-04-20/ptwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/pt/2018-04-20/ptwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/pt/2018-04-20/ptwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/pt/2018-04-20/ptwiki_20180420_300d.txt.bz2)

#### Russian

- *ruwiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/ru/2018-04-20/ruwiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/ru/2018-04-20/ruwiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/ru/2018-04-20/ruwiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/ru/2018-04-20/ruwiki_20180420_300d.txt.bz2)

#### Spanish

- *eswiki_20180420* (window=5, iteration=10, negative=15):
  [100d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/es/2018-04-20/eswiki_20180420_100d.pkl.bz2)
  [100d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/es/2018-04-20/eswiki_20180420_100d.txt.bz2)
  [300d (bin)](http://wikipedia2vec.s3.amazonaws.com/models/es/2018-04-20/eswiki_20180420_300d.pkl.bz2)
  [300d (txt)](http://wikipedia2vec.s3.amazonaws.com/models/es/2018-04-20/eswiki_20180420_300d.txt.bz2)
