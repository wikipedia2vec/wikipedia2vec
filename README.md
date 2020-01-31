Wikipedia2Vec
=============

[![Fury badge](https://badge.fury.io/py/wikipedia2vec.png)](http://badge.fury.io/py/wikipedia2vec)
[![CircleCI](https://circleci.com/gh/wikipedia2vec/wikipedia2vec.svg?style=svg)](https://circleci.com/gh/wikipedia2vec/wikipedia2vec)

Wikipedia2Vec is a tool used for obtaining embeddings (or vector representations) of words and entities (i.e., concepts that have corresponding pages in Wikipedia) from Wikipedia.
It is developed and maintained by [Studio Ousia](http://www.ousia.jp).

This tool enables you to learn embeddings of words and entities simultaneously, and places similar words and entities close to one another in a continuous vector space.
Embeddings can be easily trained by a single command with a publicly available Wikipedia dump as input.

This tool implements the [conventional skip-gram model](https://en.wikipedia.org/wiki/Word2vec) to learn the embeddings of words, and its extension proposed in [Yamada et al. (2016)](https://arxiv.org/abs/1601.01343) to learn the embeddings of entities.

An empirical comparison between Wikipedia2Vec and existing embedding tools (i.e., FastText, Gensim, RDF2Vec, and Wiki2vec) is available [here](https://arxiv.org/abs/1812.06280).

Documentation  are available online at [http://wikipedia2vec.github.io/](http://wikipedia2vec.github.io/).

## Basic Usage

Wikipedia2Vec can be installed via PyPI:

```bash
% pip install wikipedia2vec
```

With this tool, embeddings can be learned by running a *train* command with a Wikipedia dump as input.
For example, the following commands download the latest English Wikipedia dump and learn embeddings from this dump:

```bash
% wget https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2
% wikipedia2vec train enwiki-latest-pages-articles.xml.bz2 MODEL_FILE
```

Then, the learned embeddings are written to *MODEL\_FILE*.
Note that this command can take many optional parameters.
Please refer to [our documentation](https://wikipedia2vec.github.io/wikipedia2vec/commands/) for further details.

## Pretrained Embeddings

Pretrained embeddings for 12 languages (i.e., English, Arabic, Chinese, Dutch, French, German, Italian, Japanese, Polish, Portuguese, Russian, and Spanish) can be downloaded from [this page](https://wikipedia2vec.github.io/wikipedia2vec/pretrained/).

## Use Cases

Wikipedia2Vec has been used in the following applications:

* Entity linking: [Yamada et al., 2016](https://arxiv.org/abs/1601.01343), [Eshel et al., 2017](https://arxiv.org/abs/1706.09147), [Chen et al., 2019](https://arxiv.org/abs/1911.03834).
* Named entity recognition: [Sato et al., 2017](http://www.aclweb.org/anthology/I17-2017), [Lara-Clares and Garcia-Serrano, 2019](http://ceur-ws.org/Vol-2421/eHealth-KD_paper_6.pdf).
* Question answering: [Yamada et al., 2017](https://arxiv.org/abs/1803.08652).
* Entity typing: [Yamada et al., 2018](https://arxiv.org/abs/1806.02960).
* Text classification: [Yamada et al., 2018](https://arxiv.org/abs/1806.02960), [Yamada and Shindo, 2019](https://arxiv.org/abs/1909.01259).
* Paraphrase detection: [Duong et al., 2018](https://ieeexplore.ieee.org/abstract/document/8606845).
* Knowledge graph completion: [Shah et al., 2019](https://aaai.org/ojs/index.php/AAAI/article/view/4162).
* Fake news detection: [Singh et al., 2019](https://arxiv.org/abs/1906.11126).
* Plot analysis of movies: [Papalampidi et al., 2019](https://arxiv.org/abs/1908.10328).
* Enhancement of BERT using Wikipedia knowledge: [Poerner et al., 2019](https://arxiv.org/abs/1911.03681).

## References

If you use Wikipedia2Vec in a scientific publication, please cite the following papers:

Ikuya Yamada, Akari Asai, Jin Sakuma, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, Yuji Matsumoto, [Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia](https://arxiv.org/abs/1812.06280).

```bibtex
@article{yamada2020wikipedia2vec,
  title={Wikipedia2Vec: An Efficient Toolkit for Learning and Visualizing the Embeddings of Words and Entities from Wikipedia},
  author={Yamada, Ikuya and Asai, Akari and Sakuma, Jin and Shindo, Hiroyuki and Takeda, Hideaki and Takefuji, Yoshiyasu and Matsumoto, Yuji},
  journal={arXiv preprint 1812.06280v3},
  year={2020}
}
```

Ikuya Yamada, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, [Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation](https://arxiv.org/abs/1601.01343).

```bibtex
@inproceedings{yamada2016joint,
  title={Joint Learning of the Embedding of Words and Entities for Named Entity Disambiguation},
  author={Yamada, Ikuya and Shindo, Hiroyuki and Takeda, Hideaki and Takefuji, Yoshiyasu},
  booktitle={Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning},
  year={2016},
  publisher={Association for Computational Linguistics},
  doi={10.18653/v1/K16-1025},
  pages={250--259}
}
```

The text classification model implemented in [this example](https://github.com/wikipedia2vec/wikipedia2vec/tree/master/examples/text_classification) was proposed in the following paper:

Ikuya Yamada, Hiroyuki Shindo, [Neural Attentive Bag-of-Entities Model for Text Classification](https://arxiv.org/abs/1909.01259).

```bibtex
@article{yamada2019neural,
  title={Neural Attentive Bag-of-Entities Model for Text Classification},
  author={Yamada, Ikuya and Shindo, Hiroyuki},
  booktitle={Proceedings of The 23th SIGNLL Conference on Computational Natural Language Learning},
  year={2019},
  publisher={Association for Computational Linguistics}
}
```

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
