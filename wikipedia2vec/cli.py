# -*- coding: utf-8 -*-

import click
import logging
import multiprocessing

from .dictionary import Dictionary
from .link_graph import LinkGraph
from .phrase import PhraseDictionary
from .wikipedia2vec import Wikipedia2Vec
from .utils.wiki_dump_reader import WikiDumpReader


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--min-link-count', type=int, default=10)
@click.option('--min-link-prob', type=float, default=0.1)
@click.option('--lowercase/--no-lowercase', default=True)
@click.option('--max-len', default=4)
@click.option('--pool-size', type=int, default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=30)
def build_phrase_dictionary(dump_file, out_file, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    phrase_dict = PhraseDictionary.build(dump_reader, **kwargs)
    phrase_dict.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--phrase', type=click.Path())
@click.option('--lowercase/--no-lowercase', default=True)
@click.option('--min-word-count', type=int, default=10)
@click.option('--min-entity-count', type=int, default=5)
@click.option('--pool-size', type=int, default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=30)
def build_dictionary(dump_file, out_file, phrase, **kwargs):
    dump_reader = WikiDumpReader(dump_file)

    if phrase:
        phrase_dict = PhraseDictionary.load(phrase)
    else:
        phrase_dict = None

    dictionary = Dictionary.build(dump_reader, phrase_dict, **kwargs)
    dictionary.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('dictionary_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', type=int, default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=30)
def build_link_graph(dump_file, dictionary_file, out_file, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    dictionary = Dictionary.load(dictionary_file)

    link_graph = LinkGraph.build(dump_reader, dictionary, **kwargs)
    link_graph.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('dictionary_file', type=click.Path())
@click.argument('out_file', type=click.Path())
@click.option('--bundle/--no-bundle', default=True)
@click.option('--link-graph', type=click.Path())
@click.option('--dim-size', type=int, default=300)
@click.option('--init-alpha', type=float, default=0.025)
@click.option('--min-alpha', type=float, default=0.0001)
@click.option('--window', type=int, default=10)
@click.option('--links-per-page', type=int, default=10)
@click.option('--negative', type=int, default=15)
@click.option('--word-neg-power', type=float, default=0.75)
@click.option('--entity-neg-power', type=float, default=0.0)
@click.option('--sample', type=float, default=1e-4)
@click.option('--iteration', type=int, default=5)
@click.option('--pool-size', type=int, default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=100)
def train_embedding(dump_file, dictionary_file, link_graph, out_file, bundle, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    dictionary = Dictionary.load(dictionary_file)

    if link_graph:
        link_graph = LinkGraph.load(link_graph, dictionary)

    wiki2vec = Wikipedia2Vec(dictionary)
    wiki2vec.train(dump_reader, link_graph, **kwargs)

    wiki2vec.save(out_file, bundle)


@cli.command()
@click.argument('model_file', type=click.Path())
@click.argument('out_file', type=click.File(mode='w'))
@click.argument('vocab_file', type=click.File(mode='w'))
def save_word2vec_format(model_file, out_file, vocab_file):
    wiki2vec = Wikipedia2Vec.load(model_file)
    wiki2vec.save_word2vec_format(out_file, vocab_file)
