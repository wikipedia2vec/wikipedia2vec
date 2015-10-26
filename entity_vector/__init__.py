# -*- coding: utf-8 -*-

import click
import logging
import multiprocessing
import numpy as np

REAL = np.float32

from dictionary import Dictionary
from entity_vector import EntityVector


@click.group()
def cli():
    LOG_FORMAT = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.File(mode='w'))
@click.option('--min-word-count', type=int, default=5)
@click.option('--min-entity-count', type=int, default=1)
@click.option('--parallel/--no-parallel', default=True)
@click.option('--pool-size', type=int, default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=100)
def build_dictionary(dump_file, out_file, **kwargs):
    from wiki_dump_reader import WikiDumpReader

    dump_reader = WikiDumpReader(dump_file)
    dictionary = Dictionary.build(dump_reader, **kwargs)
    dictionary.save(out_file)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('dic_file', type=click.File())
@click.argument('out_file', type=click.Path(exists=False))
@click.option('--size', type=int, default=300)
@click.option('--word-alpha', type=float, default=0.025)
@click.option('--word-min-alpha', type=float, default=0.0001)
@click.option('--entity-alpha', type=float, default=0.025)
@click.option('--entity-min-alpha', type=float, default=0.0001)
@click.option('--word-window', type=int, default=10)
@click.option('--word-negative', type=int, default=30)
@click.option('--entity-negative', type=int, default=30)
@click.option('--iteration', type=int, default=10)
@click.option('--generate-links', is_flag=True)
@click.option('--parallel/--no-parallel', default=True)
@click.option('--pool-size', type=int, default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=100)
def build_embedding(dump_file, dic_file, out_file, **kwargs):
    from wiki_dump_reader import WikiDumpReader

    dump_reader = WikiDumpReader(dump_file)
    dictionary = Dictionary.load(dic_file)

    train_kwargs = dict(
        parallel=kwargs.pop('parallel'),
        pool_size=kwargs.pop('pool_size'),
        chunk_size=kwargs.pop('chunk_size')
    )

    ent_vec = EntityVector(dictionary, **kwargs)
    ent_vec.train(dump_reader, **train_kwargs)

    ent_vec.init_sims()
    ent_vec.build_vector_index()

    ent_vec.save(out_file)
