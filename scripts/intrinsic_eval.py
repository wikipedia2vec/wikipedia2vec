import os
from collections import defaultdict
from itertools import chain
from typing import Optional

import click
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
from tqdm import trange
from wikipedia2vec import Wikipedia2Vec

KORE_CATEGORIES = {
    "it_companies": ["Apple Inc.", "Google", "Facebook", "Microsoft", "IBM"],
    "celebrities": ["Angelina Jolie", "Brad Pitt", "Johnny Depp", "Jennifer Aniston", "Leonardo DiCaprio"],
    "video_games": [
        "Grand Theft Auto IV",
        "Quake (video game)",
        "Deus Ex (series)",
        "Guitar Hero (video game)",
        "Max Payne",
    ],
    "tv_series": ["The Sopranos", "The A-Team", "Futurama", "The Wire", "Mad Men"],
    "chuck_norris": ["Chuck Norris"],
}


@click.command()
@click.argument("data_dir", type=click.Path(exists=True))
@click.argument("model_file", type=click.Path(exists=True))
@click.option("-f", "--out-format", type=click.Choice(["csv", "text"]), default="text")
@click.option("--word-analogy/--no-word-analogy", default=True)
@click.option("--word-similarity/--no-word-similarity", default=True)
@click.option("--entity-similarity/--no-entity-similarity", default=True)
@click.option("--lowercase/--no-lowercase", default=True)
@click.option("--batch-size", default=1000)
@click.option("--analogy-vocab-size", default=None, type=int)
def main(
    data_dir: str,
    model_file: str,
    out_format: str,
    word_analogy: bool,
    word_similarity: bool,
    entity_similarity: bool,
    lowercase: bool,
    batch_size: int,
    analogy_vocab_size: Optional[int],
):
    model = Wikipedia2Vec.load(model_file)

    results = []

    if word_similarity:
        base_dir = os.path.join(data_dir, "word", "similarity")
        for filename in os.listdir(base_dir):
            if not filename.endswith(".txt"):
                continue

            oov_count = 0
            with open(os.path.join(base_dir, filename)) as f:
                gold = []
                estimated = []
                for line in f:
                    (w1, w2, val_str) = line.split()
                    val = float(val_str)
                    if lowercase:
                        (w1, w2) = (w1.lower(), w2.lower())
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

    if word_analogy:
        if analogy_vocab_size is None:
            target_words = [w.text for w in model.dictionary.words()]
        else:
            target_words = [
                w.text
                for w in sorted(model.dictionary.words(), key=lambda w: w.count, reverse=True)[:analogy_vocab_size]
            ]

        word_emb = np.empty((len(target_words), model.syn0.shape[1]))
        vocab = {}
        for n, word in enumerate(target_words):
            word_emb[n] = model.get_word_vector(word)
            vocab[word] = n
        word_emb = word_emb / np.linalg.norm(word_emb, 2, axis=1, keepdims=True)

        base_dir = os.path.join(data_dir, "word", "analogy")
        for filename in os.listdir(base_dir):
            with open(os.path.join(base_dir, filename)) as f:
                (A_ind, B_ind, C_ind, D_ind) = ([], [], [], [])
                oov_count = 0
                for n, line in enumerate(f):
                    if not line.startswith(":"):
                        if lowercase:
                            indices = list(map(vocab.get, line.lower().split()))
                        else:
                            indices = list(map(vocab.get, line.split()))
                        if not all(i is not None for i in indices):
                            oov_count += 1
                            continue

                        (a_ind, b_ind, c_ind, d_ind) = indices
                        A_ind.append(a_ind)
                        B_ind.append(b_ind)
                        C_ind.append(c_ind)
                        D_ind.append(d_ind)

                (A, B, C) = (word_emb[A_ind], word_emb[B_ind], word_emb[C_ind])
                D = B - A + C
                del A, B, C

                predictions = []

                for i in trange(0, D.shape[0], batch_size, desc=filename[:-4]):
                    D_batch = D[i : i + batch_size]
                    dot_ret = np.dot(word_emb, D_batch.T)
                    for j, indices in enumerate(
                        zip(A_ind[i : i + batch_size], B_ind[i : i + batch_size], C_ind[i : i + batch_size])
                    ):
                        dot_ret[indices, j] = float("-inf")
                    predictions.append(np.argmax(dot_ret, 0))

                results.append((filename[:-4], np.mean(np.hstack(predictions) == D_ind), oov_count))

    if entity_similarity:
        category_mapping = {e: c for (c, l) in KORE_CATEGORIES.items() for e in l}

        base_dir = os.path.join(data_dir, "entity", "similarity")
        for filename in os.listdir(base_dir):
            with open(os.path.join(base_dir, filename)) as f:
                if filename == "KORE.txt":
                    data = defaultdict(list)
                    title = None
                    for line in f:
                        line = line.rstrip()
                        if line.startswith("\t"):
                            data[title].append(line[1:])
                        else:
                            title = line

                    kore_results = defaultdict(list)
                    oov_count = 0
                    for title, title_list in data.items():
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

                    results.append((filename[:-4], np.mean(list(chain(*kore_results.values()))), oov_count))

                else:
                    gold = []
                    estimated = []
                    oov_count = 0
                    for n, line in enumerate(f):
                        if n == 0:
                            continue
                        line = line.rstrip()
                        (_, _, title1, _, _, title2, score) = line.split("\t")

                        try:
                            v1 = model.get_entity_vector(title1.replace("_", " "))
                        except KeyError:
                            oov_count += 1
                            continue
                        try:
                            v2 = model.get_entity_vector(title2.replace("_", " "))
                        except KeyError:
                            oov_count += 1
                            continue

                        gold.append(float(score))
                        estimated.append(1.0 - cosine(v1, v2))

                    results.append((filename[:-4], spearmanr(gold, estimated)[0], oov_count))

    if out_format == "text":
        for name, score, oov_count in results:
            print(f"{name}: ")
            print(f"  Spearman score: {score:.4f}")
            print(f"  OOV instances: {oov_count}")

    elif out_format == "csv":
        print("name," + ",".join([o[0] for o in results]))
        print("score," + ",".join(["%.4f" % o[1] for o in results]))
        print("oov," + ",".join(["%d" % o[2] for o in results]))


if __name__ == "__main__":
    main()
