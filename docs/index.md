<h1 id="main_title">Wikipedia2Vec</h1>
---

<a class="github-button" href="https://github.com/wikipedia2vec/wikipedia2vec" data-size="large" data-show-count="true" aria-label="Star wikipedia2vec/wikipedia2vec on GitHub">Star</a>

Wikipedia2Vec is a tool used for obtaining embeddings (or vector representations) of words and entities (i.e., concepts that have corresponding pages in Wikipedia) from Wikipedia.
It is developed and maintained by [Studio Ousia](http://www.ousia.jp).

This tool enables you to learn embeddings of words and entities simultaneously, and places similar words and entities close to one another in a continuous vector space.
Embeddings can be easily trained by a single command with a publicly available Wikipedia dump as input.

This tool implements the [conventional skip-gram model](https://en.wikipedia.org/wiki/Word2vec) to learn the embeddings of words, and its extension proposed in [Yamada et al. (2016)](https://arxiv.org/abs/1601.01343) to learn the embeddings of entities.
This tool has been used in several state-of-the-art NLP models such as [entity linking](https://arxiv.org/abs/1601.01343), [named entity recognition](http://www.aclweb.org/anthology/I17-2017), [knowledge graph completion](https://www.aaai.org/Papers/AAAI/2019/AAAI-ShahH.6029.pdf), [entity relatedness](https://arxiv.org/abs/1601.01343), and [question answering](https://arxiv.org/abs/1803.08652).

This tool has been tested on Linux, Windows, and macOS.

An empirical comparison between Wikipedia2Vec and existing embedding tools (i.e., FastText, Gensim, RDF2Vec, and Wiki2vec) is available [here](https://arxiv.org/abs/1812.06280).

The source code of the neural text classification model built upon Wikipedia2Vec is available [here](https://github.com/wikipedia2vec/wikipedia2vec/tree/master/examples/text_classification).

Pretrained embeddings for 12 languages (i.e., English, Arabic, Chinese, Dutch, French, German, Italian, Japanese, Polish, Portuguese, Russian, and Spanish) can be downloaded from [this page](pretrained.md).

References
----------

If you use Wikipedia2Vec in a scientific publication, please cite the following paper:

Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, [Wikipedia2Vec: An Optimized Tool for Learning Embeddings of Words and Entities from Wikipedia](https://arxiv.org/abs/1812.06280).

```bibtex
@article{yamada2018wikipedia2vec,
  title={Wikipedia2Vec: An Optimized Tool for Learning Embeddings of Words and Entities from Wikipedia},
  author={Yamada, Ikuya and Asai, Akari and Shindo, Hiroyuki and Takeda, Hideaki and Takefuji, Yoshiyasu},
  journal={arXiv preprint 1812.06280},
  year={2018}
}
```

Wikipedia2Vec is an official implementation of the embedding model proposed in the following paper:

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

The text classification model contained in [this example](https://github.com/wikipedia2vec/wikipedia2vec/tree/master/examples/text_classification) was proposed in the following paper:

Ikuya Yamada, Hiroyuki Shindo, [Neural Attentive Bag-of-Entities Model for Text Classification](https://arxiv.org/abs/1909.01259).

```bibtex
@article{yamada2019neural,
  title={Neural Attentive Bag-of-Entities Model for Text Classification},
  author={Yamada, Ikuya and Shindo, Hiroyuki},
  booktitle={Proceedings of The 20th SIGNLL Conference on Computational Natural Language Learning},
  year={2019},
  publisher={Association for Computational Linguistics}
}
```

License
-------

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
