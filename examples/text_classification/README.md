# Neural Attentive Bag-of-Entities Model for Text Classification

[https://arxiv.org/abs/1909.01259](https://arxiv.org/abs/1909.01259)

## Introduction

This directory contains the implementation of **Neural Attentive Bag-of-Entities Model (NABoE)**, a state-of-the-art neural network model that performs text classification using a set of Wikipedia entities (*bag of entities*) as input.

For each entity name in a document (e.g., *Apple*), this model first detects Wikipedia entities that may be referred to by this name (e.g., *Apple Inc.*, *Apple (food)*) using a simple dictionary-based entity detector, and then computes the feature vector of the document using the weighted average of the entity embeddings of the detected entities.
The weights are computed using a neural attention mechanism that enables the model to focus on a small subset of the entities that are less ambiguous in meaning and more relevant to the document.
Text classification is performed using a linear layer with the feature vector as input.

The code is built upon [Wikipedia2Vec](https://github.com/wikipedia2vec/wikipedia2vec) and [PyTorch](https://pytorch.org/).
The embeddings in this model are initialized using Wikipedia2Vec embeddings.
See [the paper](https://arxiv.org/abs/1909.01259) for further details.

## Results

### 20 Newsgroups

| Model | Accuracy | F1 |
| --- | --- | --- |
| NABoE | **88.1** | **87.6** |
| Bag-of-Words SVM | 79.0 | 78.3 |
| fastText ([Joulin et al., 2016](https://arxiv.org/abs/1607.01759)) | 79.4 | - |
| fastText (bigrams) ([Joulin et al., 2016](https://arxiv.org/abs/1607.01759)) | 79.7 | - |
| BoE ([Jin et al., 2016](https://www.ijcai.org/Proceedings/16/Papers/401.pdf)) | 83.1 | 82.7 |
| SWEM ([Shen et al., 2018](https://arxiv.org/abs/1805.09843)) | 85.3 | 85.5 |
| TextEnt ([Yamada et al., 2018](https://arxiv.org/abs/1806.02960)) | 84.5 | 83.9 |
| TextGCN ([Yao et al., 2019](https://arxiv.org/abs/1809.05679)) | 86.3 | - |

### R8

| Model | Accuracy | F1 |
| --- | --- | --- |
| NABoE | **97.8** | **93.0** |
| Bag-of-Words SVM | 94.7 | 85.1 |
| fastText (bigrams) ([Joulin et al., 2016](https://arxiv.org/abs/1607.01759)) | 94.7 | - |
| fastText ([Joulin et al., 2016](https://arxiv.org/abs/1607.01759)) | 96.1 | - |
| BoE ([Jin et al., 2016](https://www.ijcai.org/Proceedings/16/Papers/401.pdf)) | 96.5 | 88.6 |
| SWEM ([Shen et al., 2018](https://arxiv.org/abs/1805.09843)) | 96.7 | 89.8 |
| TextEnt ([Yamada et al., 2018](https://arxiv.org/abs/1806.02960)) | 96.7 | 91.0 |
| TextGCN ([Yao et al., 2019](https://arxiv.org/abs/1809.05679)) | 97.1 | - |

The above results are obtained by averaging the results over 10 runs.

Note that the results are slightly better than the ones reported in the original paper because we tuned hyperparameters and improved the model implementation (i.e., adding weight decay, learning rate warmup, and dropout regularization).

## Reproducing Results

The results presented above can be reproduced as follows.
**Python 3.6+** is required to run this experiment.

**Clone the repository:**

```bash
% git clone https://github.com/wikipedia2vec/wikipedia2vec.git
% cd wikipedia2vec/examples/text_classification
```

**Install required packages:**

```bash
% pip install -r requirements.txt
```

**Download Wikipedia2Vec embeddings:**

```bash
% wget https://wikipedia2vec.s3-ap-northeast-1.amazonaws.com/misc/text_classification/enwiki_20180420_lg1_300d.pkl.bz2
% bunzip2 enwiki_20180420_lg1_300d.pkl.bz2
```

**Download entity detector model:**

```bash
% wget https://wikipedia2vec.s3-ap-northeast-1.amazonaws.com/misc/text_classification/enwiki_20180420_entity_linker.pkl.bz2
% bunzip2 enwiki_20180420_entity_linker.pkl.bz2
```

**Download Reuters-21578 dataset:**

```bash
% mkdir reuters-21578
% wget http://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz
% tar xzf reuters21578.tar.gz -C reuters-21578
```

**Run experiments:**

*20 Newsgroups*

```bash
% python main.py train-classifier enwiki_20180420_lg1_300d.pkl enwiki_20180420_entity_linker.pkl --dataset=20ng
```

*R8*

```bash
% python main.py train-classifier enwiki_20180420_lg1_300d.pkl enwiki_20180420_entity_linker.pkl --dataset=r8 --dataset-path=reuters-21578
```

NOTE: You can speed up the training by specifying *--use-gpu* option if your machine has a GPU.

## Building Wikipedia2Vec Embeddings and Entity Detector

It is easy to build the Wikipedia2Vec embeddings and the entity detector used above.
First, you need to select and download a Wikipedia dump file (*enwiki-DATE-pages-articles.xml.bz2*) available at [Wikimedia Downloads](https://dumps.wikimedia.org/enwiki/).

**Train Wikipedia2Vec embeddings:**

```bash
% wikipedia2vec train WIKIPEDIA_DUMP_FILE WIKIPEDIA2VEC_FILE
```

**Build entity detector model:**

```bash
% python main.py build-dump-db WIKIPEDIA_DUMP_FILE DUMP_DB_FILE
% python main.py build-entity-linker DUMP_DB_FILE ENTITY_DETECTOR_FILE
```

These *WIKIPEDIA2VEC_FILE* and *ENTITY_DETECTOR_FILE* can be specified as arguments of `main.py`.

## Reference

```bibtex
@article{yamada2019neural,
  title={Neural Attentive Bag-of-Entities Model for Text Classification},
  author={Yamada, Ikuya and Shindo, Hiroyuki},
  booktitle={Proceedings of The 23th SIGNLL Conference on Computational Natural Language Learning},
  year={2019},
  publisher={Association for Computational Linguistics}
}
```
