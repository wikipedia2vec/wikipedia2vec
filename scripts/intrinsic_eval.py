#!/usr/bin/env python3

import click
import os
import numpy as np
from collections import defaultdict
from itertools import chain
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from tqdm import trange
from wikipedia2vec import Wikipedia2Vec

KORE_CATEGORIES = {
    'it_companies': ['Apple Inc.', 'Google', 'Facebook', 'Microsoft', 'IBM'],
    'celebrities': ['Angelina Jolie', 'Brad Pitt', 'Johnny Depp', 'Jennifer Aniston', 'Leonardo DiCaprio'],
    'video_games': ['Grand Theft Auto IV', 'Quake (video game)', 'Deus Ex (series)', 'Guitar Hero (video game)', 'Max Payne'],
    'tv_series': ['The Sopranos', 'The A-Team', 'Futurama', 'The Wire', 'Mad Men'],
    'chuck_norris': ['Chuck Norris'],
}


@click.command()
@click.argument('model_file', type=click.Path(exists=True))
@click.argument('data_dir', type=click.Path(exists=True), default='data')
@click.option('-f', '--out-format', type=click.Choice(['csv', 'text']), default='text')
@click.option('--word-analogy/--no-word-analogy', default=True)
@click.option('--word-similarity/--no-word-similarity', default=True)
@click.option('--entity-relatedness/--no-entity-relatedness', default=True)
@click.option('--batch-size', default=1000)
@click.option('--vocab-size', default=300000)
def main(data_dir, model_file, out_format, word_analogy, word_similarity, entity_relatedness,
         batch_size, vocab_size):
    model = Wikipedia2Vec.load(model_file)

    results = []

    if word_similarity:
        for filename in os.listdir(data_dir):
            if not filename.endswith('.txt'):
                continue

            if filename in ('KORE.txt', 'WikiSim-353.txt', 'EN-GOOGLE.txt'):
                continue

            oov_count = 0
            with open(os.path.join(data_dir, filename)) as f:
                gold = []
                estimated = []
                for line in f:
                    (w1, w2, val) = line.split()
                    (w1, w2, val) = w1.lower(), w2.lower(), float(val)
                    try:
                        v1 = model.get_word_vector(w1)
                    except KeyError:
                        oov_count += 1
                        continue
                    try:
                        v2 = model.get_word_vector(w2)
                    except KeyError:
                        oov_count += 1
                        continue

                    gold.append(val)
                    estimated.append(1.0 - cosine(v1, v2))

                results.append((filename[:-4], spearmanr(gold, estimated)[0], oov_count))

    if entity_relatedness:
        with open(os.path.join(data_dir, 'WikiSim-353.txt')) as f:
            gold = []
            estimated = []
            oov_count = 0
            for (n, line) in enumerate(f):
                if n == 0:
                    continue
                line = line.rstrip()
                (_, _, title1, _, _, title2, score) = line.split('\t')

                try:
                    v1 = model.get_entity_vector(title1.replace('_', ' '))
                except KeyError:
                    oov_count += 1
                    continue
                try:
                    v2 = model.get_entity_vector(title2.replace('_', ' '))
                except KeyError:
                    oov_count += 1
                    continue

                gold.append(float(score))
                estimated.append(1.0 - cosine(v1, v2))

            results.append(('WikiSim-353', spearmanr(gold, estimated)[0], oov_count))

        category_mapping = {e: c for (c, l) in KORE_CATEGORIES.items() for e in l}

        with open(os.path.join(data_dir, 'KORE.txt')) as f:
            data = defaultdict(list)
            title = None
            for line in f:
                line = line.rstrip()
                if line.startswith('\t'):
                    data[title].append(line[1:])
                else:
                    title = line

            kore_results = defaultdict(list)
            oov_count = 0
            for (title, title_list) in data.items():
                try:
                    v1 = model.get_entity_vector(title)
                except KeyError:
                    oov_count += len(title_list)
                    continue

                estimated = []
                for title2 in title_list:
                    try:
                        v2 = model.get_entity_vector(title2)
                    except KeyError:
                        oov_count += 1
                        continue
                    estimated.append(1.0 - cosine(v1, v2))

                gold = list(reversed(range(len(estimated))))
                kore_results[category_mapping[title]].append(spearmanr(gold, estimated)[0])

        results.append(('KORE', np.mean(list(chain(*kore_results.values()))), oov_count))

    if word_analogy:
        with open(os.path.join(data_dir, 'EN-GOOGLE.txt')) as f:
            (A_ind, B_ind, C_ind, D_ind) = ([], [], [], [])
            oov_count = 0
            for (n, line) in enumerate(f):
                if not line.startswith(':'):
                    words = list(map(model.get_word, line.lower().split()))
                    if not all(w is not None for w in words):
                        oov_count += 1
                        continue

                    (a_ind, b_ind, c_ind, d_ind) = map(lambda o: o.index, words)
                    A_ind.append(a_ind)
                    B_ind.append(b_ind)
                    C_ind.append(c_ind)
                    D_ind.append(d_ind)

            word_emb = model.syn0 / np.linalg.norm(model.syn0, 2, axis=1, keepdims=True)

            (A, B, C) = (word_emb[A_ind], word_emb[B_ind], word_emb[C_ind])
            D = (B - A + C)
            del A, B, C

            predictions = []

            for i in trange(0, D.shape[0], batch_size):
                D_batch = D[i:i+batch_size]
                dot_ret = np.dot(word_emb, D_batch.T)
                for (j, indices) in enumerate(zip(A_ind[i:i+batch_size], B_ind[i:i+batch_size],
                                                  C_ind[i:i+batch_size])):
                    dot_ret[indices, j] = float('-inf')
                predictions.append(np.argmax(dot_ret, 0))

            results.append(('EN-GOOGLE', np.mean(np.hstack(predictions) == D_ind), oov_count))

    if out_format == 'text':
        for (name, score, oov_count) in results:
                print('%s: ' % name)
                print('  Spearman score: %.4f' % score)
                print('  OOV instances: %d' % oov_count)

    elif out_format == 'csv':
        print('name,' + ','.join([o[0] for o in results]))
        print('score,' + ','.join(['%.4f' % o[1] for o in results]))
        print('oov,' + ','.join(['%d' % o[2] for o in results]))


if __name__ == '__main__':
    main()
