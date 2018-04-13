# -*- coding: utf-8 -*-

import click
import logging
import multiprocessing
import time

from gensim.corpora import WikiCorpus
from gensim.models import Word2Vec, Phrases
from gensim.models.word2vec import LineSentence


@click.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('corpus_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--phrase', is_flag=True)
@click.option('--size', default=100)
@click.option('--window', default=5)
@click.option('--min-count', default=5)
@click.option('--negative', default=5)
@click.option('--iter', default=5)
@click.option('--workers', default=multiprocessing.cpu_count())
def main(dump_file, corpus_file, out_file, phrase, **kwargs):
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.setLevel(level=logging.INFO)

    start_time = time.time()

    with open(corpus_file, 'w') as f:
        wiki_corpus = WikiCorpus(dump_file, lemmatize=False, dictionary={})
        for text in wiki_corpus.get_texts():
            f.write(' '.join(text) + '\n')

    corpus_time = time.time()
    print('Elapsed: %d seconds' % (corpus_time - start_time))

    sentences = LineSentence(corpus_file)
    if phrase:
        sentences = Phrases(sentences)

    model = Word2Vec(sentences, sg=1, **kwargs)
    model.save(out_file)

    now = time.time()
    print('Total: %d seconds' % (now - start_time))
    print('Preprocess: %d seconds' % (corpus_time - start_time))
    print('Train: %d seconds' % (now - corpus_time))


if __name__ == '__main__':
    main()
