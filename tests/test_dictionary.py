import pkg_resources
import unittest
from tempfile import NamedTemporaryFile
from unittest import mock

import numpy as np

from wikipedia2vec.dictionary import Dictionary, Item, Word, Entity
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.utils.tokenizer import get_tokenizer
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader


class TestWord(unittest.TestCase):
    def test_text_property(self):
        word = Word("text", 1000, 100, 10)
        self.assertEqual(word.text, "text")

    def test_index_property(self):
        word = Word("text", 1000, 100, 10)
        self.assertEqual(word.index, 1000)

    def test_count_property(self):
        word = Word("text", 1000, 100, 10)
        self.assertEqual(word.count, 100)

    def test_doc_count_property(self):
        word = Word("text", 1000, 100, 10)
        self.assertEqual(word.doc_count, 10)


class TestEntity(unittest.TestCase):
    def test_title_property(self):
        entity = Entity("Title", 1000, 100, 10)
        self.assertEqual(entity.title, "Title")

    def test_index_property(self):
        entity = Entity("Title", 1000, 100, 10)
        self.assertEqual(entity.index, 1000)

    def test_count_property(self):
        entity = Entity("Title", 1000, 100, 10)
        self.assertEqual(entity.count, 100)

    def test_doc_count_property(self):
        entity = Entity("Title", 1000, 100, 10)
        self.assertEqual(entity.doc_count, 10)


db = None
db_file = None
dictionary = None


class TestDictionary(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global db, db_file, tokenizer, dictionary
        dump_file = pkg_resources.resource_filename("tests", "test_data/enwiki-pages-articles-sample.xml.bz2")
        dump_reader = WikiDumpReader(dump_file)
        db_file = NamedTemporaryFile()

        DumpDB.build(dump_reader, db_file.name, 1, 1)
        db = DumpDB(db_file.name)

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

    @classmethod
    def tearDownClass(cls):
        db_file.close()

    def test_uuid_property(self):
        self.assertIsInstance(dictionary.uuid, str)
        self.assertEqual(32, len(dictionary.uuid))

    def test_language_property(self):
        self.assertEqual("en", dictionary.language)

    def test_lowercase_property(self):
        self.assertEqual(True, dictionary.lowercase)

    def test_build_params_property(self):
        build_params = dictionary.build_params
        self.assertEqual(build_params["dump_db"], db.uuid)
        self.assertTrue(build_params["dump_file"].endswith("enwiki-pages-articles-sample.xml.bz2"))
        self.assertEqual(2, build_params["min_word_count"])
        self.assertEqual(1, build_params["min_entity_count"])
        self.assertIsInstance(build_params["build_time"], float)
        self.assertGreater(build_params["build_time"], 0)

    def test_min_paragraph_len_property(self):
        self.assertEqual(5, dictionary.min_paragraph_len)

    def test_entity_offset_property(self):
        self.assertEqual(910, dictionary.entity_offset)

    def test_word_size_property(self):
        self.assertEqual(910, dictionary.word_size)

    def test_entity_size_property(self):
        self.assertEqual(232, dictionary.entity_size)

    def test_len(self):
        self.assertEqual(1142, len(dictionary))

    def test_iterator(self):
        items = list(dictionary)
        self.assertEqual(len(dictionary), len(items))
        for item in items:
            self.assertIsInstance(item, Item)

    def test_words_iterator(self):
        words = list(dictionary.words())
        self.assertEqual(dictionary.word_size, len(words))
        for word in words:
            self.assertIsInstance(word, Word)

    def test_entities_iterator(self):
        entities = list(dictionary.entities())
        self.assertEqual(dictionary.entity_size, len(entities))
        for entity in entities:
            self.assertIsInstance(entity, Entity)

    def test_get_word(self):
        word = dictionary.get_word("the")
        self.assertIsInstance(word, Word)
        self.assertEqual("the", word.text)
        self.assertEqual(201, word.index)
        self.assertEqual(424, word.count)
        self.assertEqual(2, word.doc_count)

    def test_get_word_not_exist(self):
        self.assertEqual(None, dictionary.get_word("foobar"))

    def test_get_word_index(self):
        self.assertEqual(201, dictionary.get_word_index("the"))

    def test_get_word_index_not_exist(self):
        self.assertEqual(-1, dictionary.get_word_index("foobar"))

    def test_get_entity(self):
        entity = dictionary.get_entity("Computer system")
        self.assertIsInstance(entity, Entity)
        self.assertEqual(1134, entity.index)
        self.assertEqual(1, entity.count)
        self.assertEqual(1, entity.doc_count)

    def test_get_entity_redirect(self):
        self.assertEqual("Computer accessibility", dictionary.get_entity("AccessibleComputing").title)
        self.assertIsNone(dictionary.get_entity("AccessibleComputing", resolve_redirect=False))

    def test_get_entity_not_exist(self):
        self.assertIsNone(dictionary.get_entity("Foo"))

    def test_get_entity_index(self):
        self.assertEqual(1134, dictionary.get_entity_index("Computer system"))

    def test_get_entity_index_not_exist(self):
        self.assertEqual(-1, dictionary.get_entity_index("Foo"))

    def test_get_item_by_index(self):
        item = dictionary.get_item_by_index(201)
        self.assertIsInstance(item, Word)
        self.assertEqual("the", item.text)
        self.assertEqual(201, item.index)
        self.assertEqual(424, item.count)
        self.assertEqual(2, item.doc_count)

        item2 = dictionary.get_item_by_index(1134)
        self.assertIsInstance(item2, Entity)
        self.assertEqual("Computer system", item2.title)
        self.assertEqual(1134, item2.index)
        self.assertEqual(1, item2.count)
        self.assertEqual(1, item2.doc_count)

    def test_get_item_by_index_not_exist(self):
        self.assertRaises(KeyError, dictionary.get_item_by_index, 100000)

    def test_get_word_by_index(self):
        word = dictionary.get_word_by_index(201)
        self.assertIsInstance(word, Word)
        self.assertEqual("the", word.text)
        self.assertEqual(201, word.index)
        self.assertEqual(424, word.count)
        self.assertEqual(2, word.doc_count)

    def test_get_word_by_index_not_exist(self):
        self.assertRaises(KeyError, dictionary.get_word_by_index, 1000)

    def test_get_entity_by_index(self):
        entity = dictionary.get_entity_by_index(1134)
        self.assertIsInstance(entity, Entity)
        self.assertEqual("Computer system", entity.title)
        self.assertEqual(1134, entity.index)
        self.assertEqual(1, entity.count)
        self.assertEqual(1, entity.doc_count)

    def test_get_entity_by_index_not_exist(self):
        self.assertRaises(KeyError, dictionary.get_entity_by_index, 0)

    def test_build_without_category(self):
        dictionary_no_category = Dictionary.build(
            db,
            tokenizer=tokenizer,
            lowercase=True,
            min_word_count=2,
            min_entity_count=1,
            min_paragraph_len=5,
            category=False,
            disambi=True,
            pool_size=1,
            chunk_size=1,
            progressbar=False,
        )
        for entity in dictionary_no_category.entities():
            self.assertFalse(entity.title.startswith("Category:"))

    def test_build_without_disambi(self):
        # treat "Computer accessibility" as a disambiguation page
        with mock.patch.object(
            DumpDB,
            "is_disambiguation",
            new_callable=lambda: lambda self, title: bool(title == "Computer accessibility"),
        ):
            dictionary_no_disambi = Dictionary.build(
                db,
                tokenizer=tokenizer,
                lowercase=True,
                min_word_count=2,
                min_entity_count=1,
                min_paragraph_len=5,
                category=True,
                disambi=False,
                pool_size=1,
                chunk_size=1,
                progressbar=False,
            )
        self.assertIsNone(dictionary_no_disambi.get_entity("Computer accessibility"))
        self.assertIsNotNone(dictionary_no_disambi.get_entity("Computer system"))

    def test_save_load(self):
        def validate(obj):
            s1 = dictionary.serialize()
            s2 = obj.serialize()
            for key in s1.keys():
                if isinstance(s1[key], np.ndarray):
                    self.assertTrue(np.array_equal(s1[key], s2[key]))
                else:
                    self.assertEqual(s1[key], s2[key])

        validate(Dictionary.load(dictionary.serialize()))
        validate(Dictionary.load(dictionary.serialize(shared_array=True)))

        with NamedTemporaryFile() as f:
            dictionary.save(f.name)
            validate(Dictionary.load(f.name))


if __name__ == "__main__":
    unittest.main()
