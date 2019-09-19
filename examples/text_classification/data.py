import functools
import logging
import os
import random
import re
import unicodedata
from collections import Counter
import numpy as np
from bs4 import BeautifulSoup
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from tqdm import tqdm

PAD_TOKEN = '<PAD>'
WHITESPACE_REGEXP = re.compile(r'\s+')

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, name, instances, label_names):
        self.name = name
        self.instances = instances
        self.label_names = label_names

    def __iter__(self):
        for instance in self.instances:
            yield instance

    def __len__(self):
        return len(self.instances)

    def get_instances(self, fold=None):
        if fold is None:
            return self.instances
        else:
            return [ins for ins in self.instances if ins.fold == fold]


class DatasetInstance(object):
    def __init__(self, text, label, fold):
        self.text = text
        self.label = label
        self.fold = fold


def generate_features(dataset, tokenizer, entity_linker, min_count, max_word_length, max_entity_length):

    @functools.lru_cache(maxsize=None)
    def tokenize(text):
        return tokenizer.tokenize(text)

    @functools.lru_cache(maxsize=None)
    def detect_mentions(text):
        return entity_linker.detect_mentions(text)

    def create_numpy_sequence(source_sequence, length, dtype):
        ret = np.zeros(length, dtype=dtype)
        source_sequence = source_sequence[:length]
        ret[:len(source_sequence)] = source_sequence
        return ret

    logger.info('Creating vocabulary...')
    word_counter = Counter()
    entity_counter = Counter()
    for instance in tqdm(dataset):
        word_counter.update(t.text for t in tokenize(instance.text) if t.text not in ENGLISH_STOP_WORDS)
        entity_counter.update(m.title for m in detect_mentions(instance.text))

    words = [word for word, count in word_counter.items() if count >= min_count]
    word_vocab = {word: index for index, word in enumerate(words, 1)}
    word_vocab[PAD_TOKEN] = 0

    entity_titles = [title for title, count in entity_counter.items() if count >= min_count]
    entity_vocab = {title: index for index, title in enumerate(entity_titles, 1)}
    entity_vocab[PAD_TOKEN] = 0

    ret = dict(train=[], dev=[], test=[], word_vocab=word_vocab, entity_vocab=entity_vocab)

    for fold in ('train', 'dev', 'test'):
        for instance in dataset.get_instances(fold):
            word_ids = [word_vocab[token.text] for token in tokenize(instance.text) if token.text in word_vocab]
            entity_ids = []
            prior_probs = []
            for mention in detect_mentions(instance.text):
                if mention.title in entity_vocab:
                    entity_ids.append(entity_vocab[mention.title])
                    prior_probs.append(mention.prior_prob)

            ret[fold].append(dict(word_ids=create_numpy_sequence(word_ids, max_word_length, np.int),
                                  entity_ids=create_numpy_sequence(entity_ids, max_entity_length, np.int),
                                  prior_probs=create_numpy_sequence(prior_probs, max_entity_length, np.float32),
                                  label=instance.label))

    return ret


def load_20ng_dataset(dev_size=0.05):
    train_data = []
    test_data = []

    for fold in ('train', 'test'):
        dataset_obj = fetch_20newsgroups(subset=fold, shuffle=False)

        for text, label in zip(dataset_obj['data'], dataset_obj['target']):
            text = normalize_text(text)
            if fold == 'train':
                train_data.append((text, label))
            else:
                test_data.append((text, label))

    dev_size = int(len(train_data) * dev_size)
    random.shuffle(train_data)

    instances = []
    instances += [DatasetInstance(text, label, 'dev') for text, label in train_data[-dev_size:]]
    instances += [DatasetInstance(text, label, 'train') for text, label in train_data[:-dev_size]]
    instances += [DatasetInstance(text, label, 'test') for text, label in test_data]

    return Dataset('20ng', instances, fetch_20newsgroups()['target_names'])


def load_r8_dataset(dataset_path, dev_size=0.05):
    label_names = ['grain', 'earn', 'interest', 'acq', 'trade', 'crude', 'ship', 'money-fx']
    label_index = {t: i for i, t in enumerate(label_names)}

    train_data = []
    test_data = []

    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.sgm'):
            with open(os.path.join(dataset_path, file_name), encoding='ISO-8859-1') as f:
                for node in BeautifulSoup(f.read(), 'html.parser').find_all('reuters'):
                    text = normalize_text(node.find('text').text)
                    label_nodes = [n.text for n in node.topics.find_all('d')]
                    if len(label_nodes) != 1:
                        continue

                    labels = [label_index[l] for l in label_nodes if l in label_index]
                    if len(labels) == 1:
                        if node['topics'] != 'YES':
                            continue
                        if node['lewissplit'] == 'TRAIN':
                            train_data.append((text, labels[0]))
                        elif node['lewissplit'] == 'TEST':
                            test_data.append((text, labels[0]))
                        else:
                            continue

    dev_size = int(len(train_data) * dev_size)
    random.shuffle(train_data)

    instances = []
    instances += [DatasetInstance(text, label, 'dev') for text, label in train_data[-dev_size:]]
    instances += [DatasetInstance(text, label, 'train') for text, label in train_data[:-dev_size]]
    instances += [DatasetInstance(text, label, 'test') for text, label in test_data]

    return Dataset('r8', instances, label_names)


def normalize_text(text):
    text = text.lower()
    text = re.sub(WHITESPACE_REGEXP, ' ', text)

    # remove accents: https://stackoverflow.com/a/518232
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = unicodedata.normalize('NFC', text)

    return text
