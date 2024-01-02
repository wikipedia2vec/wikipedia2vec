import pkg_resources
import unittest
from tempfile import NamedTemporaryFile

import numpy as np

from wikipedia2vec.dictionary import Dictionary, Entity
from wikipedia2vec.dump_db import DumpDB
from wikipedia2vec.link_graph import LinkGraph
from wikipedia2vec.utils.tokenizer import get_default_tokenizer
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader

db = None
db_file = None
dictionary = None
link_graph = None


class TestLinkGraph(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global db, db_file, dictionary, link_graph

        dump_file = pkg_resources.resource_filename("tests", "test_data/enwiki-pages-articles-sample.xml.bz2")
        dump_reader = WikiDumpReader(dump_file)
        db_file = NamedTemporaryFile()

        DumpDB.build(dump_reader, db_file.name, 1, 1)
        db = DumpDB(db_file.name)

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
        link_graph = LinkGraph.build(db, dictionary, pool_size=1, chunk_size=1, progressbar=False)

    @classmethod
    def tearDownClass(cls):
        db_file.close()

    def test_uuid_property(self):
        self.assertIsInstance(link_graph.uuid, str)
        self.assertEqual(32, len(link_graph.uuid))

    def test_build_params_property(self):
        build_params = link_graph.build_params
        self.assertEqual(build_params["dump_db"], db.uuid)
        self.assertTrue(build_params["dump_file"].endswith("enwiki-pages-articles-sample.xml.bz2"))
        self.assertIsInstance(build_params["build_time"], float)
        self.assertGreater(build_params["build_time"], 0)

    def test_neighbors(self):
        entity = dictionary.get_entity("Computer accessibility")
        neighbors = link_graph.neighbors(entity)
        self.assertEqual(113, len(neighbors))
        for entity in neighbors:
            self.assertIsInstance(entity, Entity)

        self.assertIn("Accessibility", [e.title for e in neighbors])

    def test_neighbor_indices(self):
        index = dictionary.get_entity_index("Computer accessibility")
        neighbor_indices = link_graph.neighbor_indices(index)
        self.assertIsInstance(neighbor_indices, np.ndarray)
        self.assertEqual(113, neighbor_indices.size)

        self.assertIn(dictionary.get_entity_index("Accessibility"), neighbor_indices)

    def test_save_load(self):
        def validate(obj):
            s1 = link_graph.serialize()
            s2 = obj.serialize()
            for key in s1.keys():
                if isinstance(s1[key], np.ndarray):
                    np.array_equal(s1[key], s2[key])
                else:
                    self.assertEqual(s1[key], s2[key])

        validate(LinkGraph.load(link_graph.serialize(), dictionary))
        validate(LinkGraph.load(link_graph.serialize(shared_array=True), dictionary))

        with NamedTemporaryFile() as f:
            link_graph.save(f.name)
            validate(LinkGraph.load(f.name, dictionary))


if __name__ == "__main__":
    unittest.main()
