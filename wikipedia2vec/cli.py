# -*- coding: utf-8 -*-

import click
import functools
import logging
import multiprocessing
import os

from .dictionary import Dictionary
from .link_graph import LinkGraph
from .phrase import PhraseDictionary
from .wikipedia2vec import Wikipedia2Vec
from .utils.wiki_dump_reader import WikiDumpReader

logger = logging.getLogger(__name__)


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


def common_options(func):
    @click.option('--pool-size', type=int, default=multiprocessing.cpu_count())
    @click.option('--chunk-size', type=int, default=100)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def build_phrase_dictionary_options(func):
    @click.option('--min-link-count', type=int, default=10)
    @click.option('--min-link-prob', type=float, default=0.1)
    @click.option('--max-phrase-len', default=4)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def build_dictionary_options(func):
    @click.option('--min-word-count', type=int, default=10)
    @click.option('--min-entity-count', type=int, default=5)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


def train_embedding_options(func):
    @click.option('--dim-size', type=int, default=100)
    @click.option('--init-alpha', type=float, default=0.025)
    @click.option('--min-alpha', type=float, default=0.0001)
    @click.option('--window', type=int, default=5)
    @click.option('--links-per-page', type=int, default=10)
    @click.option('--negative', type=int, default=15)
    @click.option('--word-neg-power', type=float, default=0.75)
    @click.option('--entity-neg-power', type=float, default=0.0)
    @click.option('--sample', type=float, default=1e-4)
    @click.option('--iteration', type=int, default=5)
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--lowercase/--no-lowercase', default=True)
@click.option('--phrase/--no-phrase', default=True)
@click.option('--link-graph/--no-link-graph', default=True)
@build_phrase_dictionary_options
@build_dictionary_options
@train_embedding_options
@common_options
@click.pass_context
def train(ctx, out_file, phrase, link_graph, **kwargs):
    (out_name, out_ext) = os.path.splitext(os.path.basename(out_file))
    out_path = os.path.dirname(out_file)

    dictionary_file = os.path.join(out_path, out_name + '_dic.pkl')

    def invoke(cmd, **cmd_kwargs):
        for param in cmd.params:
            if param.name not in cmd_kwargs:
                cmd_kwargs[param.name] = kwargs[param.name]
        ctx.invoke(cmd, **cmd_kwargs)

    if phrase:
        logger.info('Starting to build a phrase dictionary...')
        phrase_file = os.path.join(out_path, out_name + '_phrase.pkl')
        invoke(build_phrase_dictionary, out_file=phrase_file)

        logger.info('Starting to build a dictionary...')
        invoke(build_dictionary, out_file=dictionary_file, phrase=phrase_file)
    else:
        logger.info('Starting to build a dictionary...')
        invoke(build_dictionary, out_file=dictionary_file, phrase=None)

    if link_graph:
        logger.info('Starting to build a link graph...')
        link_graph_file = os.path.join(out_path, out_name + '_lg.pkl')
        invoke(build_link_graph, dictionary_file=dictionary_file, out_file=link_graph_file)

        logger.info('Starting to train embeddings...')
        invoke(train_embedding, dictionary_file=dictionary_file,
               link_graph=link_graph_file, out_file=out_file)
    else:
        logger.info('Starting to train embeddings...')
        invoke(train_embedding, dictionary_file=dictionary_file, out_file=out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--lowercase/--no-lowercase', default=True)
@build_phrase_dictionary_options
@common_options
def build_phrase_dictionary(dump_file, out_file, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    phrase_dict = PhraseDictionary.build(dump_reader, **kwargs)
    phrase_dict.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--phrase', type=click.Path(exists=True))
@click.option('--lowercase/--no-lowercase', default=True)
@build_dictionary_options
@common_options
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
@click.argument('dictionary_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@common_options
def build_link_graph(dump_file, dictionary_file, out_file, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    dictionary = Dictionary.load(dictionary_file)

    link_graph = LinkGraph.build(dump_reader, dictionary, **kwargs)
    link_graph.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('dictionary_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--link-graph', type=click.Path(exists=True))
@train_embedding_options
@common_options
def train_embedding(dump_file, dictionary_file, link_graph, out_file, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    dictionary = Dictionary.load(dictionary_file)

    if link_graph:
        link_graph = LinkGraph.load(link_graph, dictionary)

    wiki2vec = Wikipedia2Vec(dictionary)
    wiki2vec.train(dump_reader, link_graph, **kwargs)

    wiki2vec.save(out_file)


@cli.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
def save_text(model_file, out_file):
    wiki2vec = Wikipedia2Vec.load(model_file)
    wiki2vec.save_text(out_file)
