import logging
import multiprocessing
import random
import click
import numpy as np
import torch
from wikipedia2vec.dictionary import Dictionary
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.tokenizer.regexp_tokenizer import RegexpTokenizer
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec import Wikipedia2Vec

from data import normalize_text, load_20ng_dataset, load_r8_dataset
from entity_linker import EntityLinker
from train import train


@click.group()
@click.option('--verbose', is_flag=True)
@click.option('--seed', type=int, default=0)
def cli(verbose, seed):
    fmt = '[%(asctime)s] [%(levelname)s] %(message)s (%(funcName)s@%(filename)s:%(lineno)s)'
    if verbose:
        logging.basicConfig(level=logging.DEBUG, format=fmt)
    else:
        logging.basicConfig(level=logging.INFO, format=fmt)

    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@cli.command()
@click.argument('dump_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', type=int, default=100)
def build_dump_db(dump_file, out_file, **kwargs):
    dump_reader = WikiDumpReader(dump_file)
    DumpDB.build(dump_reader, out_file, preprocess_func=normalize_text, **kwargs)


@cli.command()
@click.argument('dump_db_file', type=click.Path(exists=True))
@click.argument('out_file', type=click.Path())
@click.option('--max-mention-length', default=30)
@click.option('--min-link-prob', default=0.01)
@click.option('--min-prior-prob', default=0.03)
@click.option('--min-link-count', default=1)
@click.option('--pool-size', default=multiprocessing.cpu_count())
@click.option('--chunk-size', default=30)
def build_entity_linker(dump_db_file, **kwargs):
    dump_db = DumpDB(dump_db_file)
    tokenizer = RegexpTokenizer()
    EntityLinker.build(dump_db, tokenizer, **kwargs)


@cli.command()
@click.argument('wikipedia2vec_file', type=click.Path(exists=True, dir_okay=False))
@click.argument('entity_linker_file', type=click.Path(exists=True, dir_okay=False))
@click.option('--dataset', type=click.Choice(['20ng', 'r8']), default='20ng')
@click.option('--dataset-path', type=click.Path(exists=True, file_okay=False))
@click.option('--dev-size', default=0.05)
@click.option('--min-count', default=3)
@click.option('--max-word-length', default=100)
@click.option('--max-entity-length', default=200)
@click.option('--batch-size', default=32)
@click.option('--patience', default=10)
@click.option('--learning-rate', default=1e-3)
@click.option('--weight-decay', default=0.1)
@click.option('--warmup-epochs', default=5)
@click.option('--dropout-prob', default=0.5)
@click.option('--use-gpu', is_flag=True)
@click.option('--use-word/--no-word', default=True)
def train_classifier(wikipedia2vec_file, entity_linker_file, dataset, dataset_path, dev_size, **kwargs):
    if dataset == '20ng':
        dataset = load_20ng_dataset(dev_size)
    else:
        dataset = load_r8_dataset(dataset_path, dev_size)

    tokenizer = RegexpTokenizer()
    entity_linker = EntityLinker(entity_linker_file)
    embedding = Wikipedia2Vec.load(wikipedia2vec_file)

    train(dataset, embedding, tokenizer, entity_linker, **kwargs)


if __name__ == '__main__':
    cli()
