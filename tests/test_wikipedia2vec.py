import os
import pkg_resources
import unittest
from collections import Counter
from tempfile import TemporaryDirectory

import numpy as np
from scipy.special import expit

from wikipedia2vec.dictionary import Dictionary, Word, Entity
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.tokenizer import get_tokenizer
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec.wikipedia2vec import Wikipedia2Vec, ItemWithScore


db = None
db_dir = None
dictionary = None
wiki2vec = None


class TestWikipedia2Vec(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global db, db_dir, tokenizer, dictionary, wiki2vec
        dump_file = pkg_resources.resource_filename("tests", "test_data/enwiki-pages-articles-sample.xml.bz2")
        dump_reader = WikiDumpReader(dump_file)
        db_dir = TemporaryDirectory()
        db_file = os.path.join(db_dir.name, "test.db")

        DumpDB.build(dump_reader, db_file, 1, 1)
        db = DumpDB(db_file)

        tokenizer = get_tokenizer("regexp")
        dictionary = Dictionary.build(
            db,
            tokenizer=tokenizer,
            lowercase=True,
            min_word_count=2,
            min_entity_count=1,
            min_paragraph_len=5,
            category=True,
            disambi=True,
            pool_size=1,
            chunk_size=1,
            progressbar=False,
        )
        wiki2vec = Wikipedia2Vec(dictionary)
        wiki2vec.syn0 = np.random.rand(len(dictionary), 100).astype(np.float32)
        wiki2vec.syn1 = np.random.rand(len(dictionary), 100).astype(np.float32)

    @classmethod
    def tearDownClass(cls):
        db.close()
        db_dir.cleanup()

    def test_dictionary_property(self):
        self.assertEqual(wiki2vec.dictionary, dictionary)

    def test_get_word(self):
        word = wiki2vec.get_word("the")
        self.assertIsInstance(word, Word)

    def test_get_word_not_exist(self):
        self.assertEqual(None, wiki2vec.get_word("foobar"))

    def test_get_entity(self):
        entity = wiki2vec.get_entity("Computer system")
        self.assertIsInstance(entity, Entity)

    def test_get_entity_not_exist(self):
        self.assertIsNone(wiki2vec.get_entity("Foo"))

    def test_get_word_vector(self):
        vector = wiki2vec.get_word_vector("the")
        self.assertEqual((100,), vector.shape)
        self.assertTrue((vector == wiki2vec.syn0[dictionary.get_word("the").index]).all())

    def test_get_word_vector_not_exist(self):
        self.assertRaises(KeyError, wiki2vec.get_word_vector, "foobar")

    def test_get_entity_vector(self):
        vector = wiki2vec.get_entity_vector("Computer system")
        self.assertEqual((100,), vector.shape)
        self.assertTrue((wiki2vec.syn0[dictionary.get_entity("Computer system").index] == vector).all())

    def test_get_entity_vector_not_exist(self):
        self.assertRaises(KeyError, wiki2vec.get_entity_vector, "Foo")

    def test_get_vector(self):
        word = dictionary.get_word("the")
        vector = wiki2vec.get_vector(word)
        self.assertEqual((100,), vector.shape)
        self.assertTrue((vector == wiki2vec.syn0[dictionary.get_word("the").index]).all())

    def test_most_similar(self):
        word = dictionary.get_word("the")
        vector = wiki2vec.syn0[word.index]
        all_scores = np.dot(wiki2vec.syn0, vector) / np.linalg.norm(wiki2vec.syn0, axis=1) / np.linalg.norm(vector)
        indexes = np.argsort(-all_scores)[:10].tolist()
        scores = [float(all_scores[index]) for index in indexes]

        ret = wiki2vec.most_similar(word, 10)
        for entry in ret:
            self.assertIsInstance(entry, ItemWithScore)
        self.assertEqual(indexes, [o.item.index for o in ret])
        self.assertEqual(scores, [o.score for o in ret])

    def test_most_similar_by_vector(self):
        word = dictionary.get_word("the")
        vector = wiki2vec.syn0[word.index]
        all_scores = np.dot(wiki2vec.syn0, vector) / np.linalg.norm(wiki2vec.syn0, axis=1) / np.linalg.norm(vector)
        indexes = np.argsort(-all_scores)[:10].tolist()
        scores = [float(all_scores[index]) for index in indexes]

        ret = wiki2vec.most_similar_by_vector(vector, 10)
        for entry in ret:
            self.assertIsInstance(entry, ItemWithScore)
        self.assertEqual(indexes, [o.item.index for o in ret])
        self.assertEqual(scores, [o.score for o in ret])

    def test_save_load(self):
        with TemporaryDirectory() as dir_name:
            file_name = os.path.join(dir_name, "model.pkl")
            wiki2vec.save(file_name)
            wiki2vec2 = Wikipedia2Vec.load(file_name, numpy_mmap_mode=None)
            self.assertTrue(np.array_equal(wiki2vec.syn0, wiki2vec2.syn0))
            self.assertTrue(np.array_equal(wiki2vec.syn1, wiki2vec2.syn1))

            serialized_dictionary = dictionary.serialize()
            serialized_dictionary2 = wiki2vec2.dictionary.serialize()
            for key in serialized_dictionary.keys():
                if isinstance(serialized_dictionary[key], np.ndarray):
                    self.assertTrue(np.array_equal(serialized_dictionary[key], serialized_dictionary2[key]))
                else:
                    self.assertEqual(serialized_dictionary[key], serialized_dictionary2[key])

    def test_save_load_text(self):
        for out_format in ("word2vec", "glove", "default"):
            with TemporaryDirectory() as dir_name:
                file_name = os.path.join(dir_name, "model.txt")
                wiki2vec.save_text(file_name, out_format=out_format)
                with open(file_name) as f:
                    if out_format == "word2vec":
                        first_line = next(f)
                        self.assertEqual(str(len(dictionary)) + " " + "100", first_line.rstrip())

                    num_items = 0
                    for line in f:
                        if out_format in ("word2vec", "glove"):
                            name, *vec_str = line.rstrip().split(" ")
                            name = name.replace("_", " ")
                        else:
                            name, vec_str = line.rstrip().split("\t")
                            vec_str = vec_str.split(" ")

                        vector = np.array([float(s) for s in vec_str], dtype=np.float32)

                        if name.startswith("ENTITY/"):
                            name = name[7:]
                            orig_vector = wiki2vec.get_entity_vector(name)
                        else:
                            orig_vector = wiki2vec.get_word_vector(name)
                        self.assertTrue(np.allclose(orig_vector, vector, atol=1e-3))

                        num_items += 1

                    self.assertEqual(len(dictionary), num_items)

                wiki2vec2 = Wikipedia2Vec.load_text(file_name)
                for word in dictionary.words():
                    self.assertTrue(
                        np.allclose(
                            wiki2vec.get_word_vector(word.text), wiki2vec2.get_word_vector(word.text), atol=1e-3
                        )
                    )
                for entity in dictionary.entities():
                    self.assertTrue(
                        np.allclose(
                            wiki2vec.get_entity_vector(entity.title),
                            wiki2vec2.get_entity_vector(entity.title),
                            atol=1e-3,
                        )
                    )
                self.assertEqual(len(dictionary), len(wiki2vec2.dictionary))

    def test_build_sampling_table(self):
        table = wiki2vec._build_word_sampling_table(0.01)
        self.assertIsInstance(table, np.ndarray)
        self.assertEqual(np.uint32, table.dtype)

        total_count = sum(word.count for word in dictionary.words())
        threshold = 0.01 * total_count
        uint_max = np.iinfo(np.uint32).max
        for word in dictionary.words():
            if word.count > total_count * 0.01:
                self.assertAlmostEqual(
                    min(1.0, (np.sqrt(word.count / threshold) + 1) * (threshold / word.count)) * uint_max,
                    table[word.index],
                    delta=1,
                )
            else:
                self.assertEqual(uint_max, table[word.index])

    def test_build_unigram_neg_table(self):
        power = 0.75
        table_size = 100000
        table = wiki2vec._build_unigram_neg_table(dictionary.words(), power=power, table_size=table_size)
        denom = sum(word.count**power for word in dictionary.words())
        for word_index, count in Counter(table).items():
            self.assertAlmostEqual(
                (dictionary.get_word_by_index(int(word_index)).count ** power) / denom,
                count / table_size,
                delta=0.0001,
            )

    def test_build_uniform_neg_table(self):
        table = wiki2vec._build_uniform_neg_table(dictionary.words())
        self.assertEqual(dictionary.word_size, table.shape[0])
        counter = Counter(table)
        for word in dictionary.words():
            self.assertEqual(1, counter[word.index])

    def test_build_exp_table(self):
        max_exp = 6
        table_size = 1000
        exp_table = wiki2vec._build_exp_table(max_exp, table_size)
        for value in range(-max_exp, max_exp):
            index = int((value + max_exp) * (table_size / max_exp / 2))
            self.assertAlmostEqual(expit(value), exp_table[index], delta=1e-2)


if __name__ == "__main__":
    unittest.main()
