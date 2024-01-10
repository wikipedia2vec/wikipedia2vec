import os
import pkg_resources
import unittest
from tempfile import TemporaryDirectory

from wikipedia2vec.dictionary import Dictionary
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.mention_db import Mention, MentionDB
from wikipedia2vec.utils.tokenizer import get_default_tokenizer
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

db = None
db_dir = None
tokenizer = None
dictionary = None


def setUpModule():
    global db, db_dir, tokenizer, dictionary

    dump_file = pkg_resources.resource_filename("tests", "test_data/enwiki-pages-articles-sample.xml.bz2")
    dump_reader = WikiDumpReader(dump_file)
    db_dir = TemporaryDirectory()
    db_file = os.path.join(db_dir.name, "test.db")

    DumpDB.build(dump_reader, db_file, 1, 1)
    db = DumpDB(db_file)

    tokenizer = get_default_tokenizer("en")
    dictionary = Dictionary.build(
        db,
        tokenizer=tokenizer,
        lowercase=True,
        min_word_count=2,
        min_entity_count=1,
        min_paragraph_len=5,
        category=False,
        disambi=False,
        pool_size=1,
        chunk_size=1,
        progressbar=False,
    )


def tearDownModule():
    db.close()
    db_dir.cleanup()


class TestMention(unittest.TestCase):
    def setUp(self):
        self._entity = dictionary.get_entity("Computer accessibility")
        self._mention = Mention(
            dictionary,
            text="mention text",
            index=self._entity.index,
            link_count=5,
            total_link_count=10,
            doc_count=100,
            start=3,
            end=6,
        )

    def test_text_property(self):
        self.assertEqual("mention text", self._mention.text)

    def test_index_property(self):
        self.assertEqual(dictionary.get_entity("Computer accessibility").index, self._mention.index)

    def test_count_properties(self):
        self.assertEqual(5, self._mention.link_count)
        self.assertEqual(10, self._mention.total_link_count)
        self.assertEqual(100, self._mention.doc_count)

    def test_start_and_end_properies(self):
        self.assertEqual(3, self._mention.start)
        self.assertEqual(6, self._mention.end)

    def test_entity_property(self):
        self.assertEqual("Computer accessibility", self._mention.entity.title)

    def test_link_prob_property(self):
        self.assertAlmostEqual(0.1, self._mention.link_prob)

    def test_prior_prob_property(self):
        self.assertAlmostEqual(0.5, self._mention.prior_prob)


mention_db = None


class TestMentionDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global mention_db
        mention_db = MentionDB.build(
            dump_db=db,
            dictionary=dictionary,
            tokenizer=tokenizer,
            min_link_prob=0.01,
            min_prior_prob=0.01,
            max_mention_len=5,
            case_sensitive=False,
            pool_size=1,
            chunk_size=1,
            progressbar=False,
        )

    def test_iterator(self):
        for mention in mention_db:
            self.assertIsInstance(mention, Mention)
            # print(mention.entity.title, mention.text)

    def test_query(self):
        mentions = mention_db.query("web")
        self.assertEqual(1, len(mentions))
        self.assertEqual(mentions[0].text, "web")
        self.assertEqual(mentions[0].entity.title, "World Wide Web")

    def test_prefix_search(self):
        self.assertEqual(mention_db.prefix_search("web service", start=0), ["web"])
        self.assertEqual(mention_db.prefix_search("this is a web service", start=0), [])
        self.assertEqual(mention_db.prefix_search("this is a web service", start=10), ["web"])

    def test_detect_mentions(self):
        text = "this is a web service"
        tokens = tokenizer.tokenize(text)
        mentions = mention_db.detect_mentions(text, tokens)
        self.assertEqual(1, len(mentions))
        self.assertEqual(mentions[0].text, "web")
        self.assertEqual(mentions[0].entity.title, "World Wide Web")

    def test_uuid_property(self):
        self.assertIsInstance(mention_db.uuid, str)
        self.assertEqual(32, len(mention_db.uuid))

    def test_build_params_property(self):
        build_params = mention_db.build_params
        self.assertEqual(build_params["dump_db"], db.uuid)
        self.assertTrue(build_params["dump_file"].endswith("enwiki-pages-articles-sample.xml.bz2"))
        self.assertIsInstance(build_params["build_time"], float)
        self.assertGreater(build_params["build_time"], 0)

    def test_save_load(self):
        def validate(obj):
            s1 = mention_db.serialize()
            s2 = obj.serialize()
            for key in s1.keys():
                self.assertEqual(s1[key], s2[key])

        validate(MentionDB.load(mention_db.serialize(), dictionary))

        with TemporaryDirectory() as dir_name:
            file_name = os.path.join(dir_name, "mention.pkl")
            mention_db.save(file_name)
            validate(MentionDB.load(file_name, dictionary))


if __name__ == "__main__":
    unittest.main()
