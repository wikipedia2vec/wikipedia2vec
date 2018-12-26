Wikipedia2Vec
=============

[![Fury badge](https://badge.fury.io/py/wikipedia2vec.png)](http://badge.fury.io/py/wikipedia2vec)
[![CircleCI](https://circleci.com/gh/wikipedia2vec/wikipedia2vec.svg?style=svg)](https://circleci.com/gh/wikipedia2vec/wikipedia2vec)

Wikipedia2Vec is a tool used for obtaining embeddings (vector representations) of words and entities from Wikipedia.
It is developed and maintained by [Studio Ousia](http://www.ousia.jp).

This tool enables you to learn embeddings of words and entities simultaneously, and places similar words and entities close to one another in a continuous vector space.
Embeddings can be easily trained by a single command with a publicly available Wikipedia dump as input.
This tool has been used in several state-of-the-art NLP models such as [entity linking](https://arxiv.org/abs/1601.01343), [named entity recognition](http://www.aclweb.org/anthology/I17-2017), [entity relatedness](https://arxiv.org/abs/1601.01343), and [question answering](https://arxiv.org/abs/1803.08652).

Documentation and pretrained embeddings are available online at [http://wikipedia2vec.github.io/](http://wikipedia2vec.github.io/).

Reference
---------

If you use Wikipedia2Vec in a scientific publication, please cite the following paper:

Ikuya Yamada, Akari Asai, Hiroyuki Shindo, Hideaki Takeda, Yoshiyasu Takefuji, [Wikipedia2Vec: An Optimized Tool for Learning Embeddings of Words and Entities from Wikipedia](https://arxiv.org/abs/1812.06280).

```text
@article{yamada2018wikipedia2vec,
  title={Wikipedia2Vec: An Optimized Tool for Learning Embeddings of Words and Entities from Wikipedia},
  author={Yamada, Ikuya and Asai, Akari and Shindo, Hiroyuki and Takeda, Hideaki and Takefuji, Yoshiyasu},
  journal={arXiv preprint 1812.06280},
  year={2018}
}
```

License
-------

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
