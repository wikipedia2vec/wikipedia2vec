import pkg_resources
import unittest

from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec.utils.wiki_page import WikiPage


class TestWikiPage(unittest.TestCase):
    def setUp(self):
        sample_dump_file = pkg_resources.resource_filename("tests", "test_data/enwiki-pages-articles-sample.xml.bz2")
        self.dump_reader = WikiDumpReader(sample_dump_file)
        self.pages = list(self.dump_reader)

    def test_is_redirect(self):
        self.assertTrue(self.pages[0].is_redirect)
        self.assertFalse(self.pages[1].is_redirect)

    def test_is_disambiguation(self):
        self.assertFalse(self.pages[0].is_disambiguation)
        wiki_page = WikiPage("Title", "en", "Text", None)
        self.assertFalse(wiki_page.is_disambiguation)
        wiki_page = WikiPage("Title", "en", "{{disambig}}", None)
        self.assertTrue(wiki_page.is_disambiguation)
        wiki_page = WikiPage("Title", "en", "text\n{{ disambig | arg }}\ntext", None)
        self.assertTrue(wiki_page.is_disambiguation)
        wiki_page = WikiPage("Title", "ja", "text\n{{   aimai   | arg1 | arg2 }}\ntext", None)
        self.assertTrue(wiki_page.is_disambiguation)

    def test_redirect(self):
        self.assertEqual("Computer accessibility", self.pages[0].redirect)

    def test_title_property(self):
        self.assertEqual("Computer accessibility", self.pages[1].title)

    def test_language_property(self):
        self.assertEqual("en", self.pages[1].language)

    def test_wiki_text_property(self):
        self.assertEqual(24949, len(self.pages[1].wiki_text))
        self.assertTrue("In [[human–computer interaction]], '''computer accessibility'''" in self.pages[1].wiki_text)


if __name__ == "__main__":
    unittest.main()
