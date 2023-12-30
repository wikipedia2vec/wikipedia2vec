import pkg_resources
import unittest

from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec.utils.wiki_page import WikiPage


class TestWikiDumpReader(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename("tests", "test_data/enwiki-pages-articles-sample.xml.bz2")
        self.dump_reader = WikiDumpReader(sample_dump_file)

    def test_dump_file_property(self):
        self.assertTrue(self.dump_reader.dump_file.endswith("enwiki-pages-articles-sample.xml.bz2"))

    def test_language_property(self):
        self.assertEqual("en", self.dump_reader.language)

    def test_iterator(self):
        pages = list(self.dump_reader)
        self.assertEqual(3, len(pages))
        for page in pages:
            self.assertIsInstance(page, WikiPage)


if __name__ == "__main__":
    unittest.main()
