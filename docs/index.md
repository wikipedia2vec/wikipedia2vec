<h1 id="main_title">Wikipedia2Vec</h1>
---

<a class="github-button" href="https://github.com/wikipedia2vec/wikipedia2vec" data-size="large" data-show-count="true" aria-label="Star wikipedia2vec/wikipedia2vec on GitHub">Star</a>

Introduction
------------

Wikipedia2Vec is a tool used for obtaining embeddings (vector representations) of words and entities from Wikipedia.
It is developed and maintained by [Studio Ousia](http://www.ousia.jp).

This tool enables you to learn embeddings that map words and entities into a unified continuous vector space.
The embeddings can be used as word embeddings, entity embeddings, and the unified embeddings of words and entities.
They are used in the state-of-the-art models of various tasks such as [entity linking](https://arxiv.org/abs/1601.01343), [named entity recognition](http://www.aclweb.org/anthology/I17-2017), [entity relatedness](https://arxiv.org/abs/1601.01343), and [question answering](https://arxiv.org/abs/1803.08652).

The embeddings can be easily trained by a single command with a publicly available Wikipedia dump as input.
The code is implemented in Python, and optimized using Cython and BLAS.

Pretrained embeddings for 12 languages can be downloaded from [this page](pretrained.md).

Reference
---------

If you use Wikipedia2Vec in a scientific publication, please cite the following paper:

Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, [Wikipedia2Vec: An Optimized Implementation for Learning Embeddings from Wikipedia](https://arxiv.org/abs/1812.06280).

```text
@article{yamada2018wikipedia2vec,
  title={Wikipedia2Vec: An Optimized Implementation for Learning Embeddings from Wikipedia},
  author={Yamada, Ikuya and Asai, Akari and Shindo, Hiroyuki and Takeda, Hideaki and Takefuji, Yoshiyasu},
  journal={arXiv preprint 1812.06280},
  year={2018}
}
```

License
-------

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
