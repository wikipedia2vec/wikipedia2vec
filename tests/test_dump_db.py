import pickle
import pkg_resources
import unittest
import zlib
from tempfile import NamedTemporaryFile

from wikipedia2vec import dump_db
from wikipedia2vec.dump_db import DumpDB, Paragraph, WikiLink
from wikipedia2vec.utils.wiki_dump_reader import WikiDumpReader
from wikipedia2vec.utils.wiki_page import WikiPage


class TestParagraph(unittest.TestCase):
    def test_text_property(self):
        paragraph = Paragraph("paragraph text", [], False)
        self.assertEqual("paragraph text", paragraph.text)

    def test_wiki_link_property(self):
        wiki_link = WikiLink("Title", "link text", 0, 3)
        paragraph = Paragraph("paragraph text", [wiki_link], False)
        self.assertEqual([wiki_link], paragraph.wiki_links)

    def test_abstract_property(self):
        paragraph = Paragraph("paragraph text", [], True)
        self.assertTrue(paragraph.abstract)
        paragraph = Paragraph("paragraph text", [], False)
        self.assertFalse(paragraph.abstract)


class TestWikiLink(unittest.TestCase):
    def test_title_property(self):
        wiki_link = WikiLink("WikiTitle", "link text", 0, 3)
        self.assertEqual("WikiTitle", wiki_link.title)

    def test_text_property(self):
        wiki_link = WikiLink("WikiTitle", "link text", 0, 3)
        self.assertEqual("link text", wiki_link.text)

    def test_start_property(self):
        wiki_link = WikiLink("WikiTitle", "link text", 0, 3)
        self.assertEqual(0, wiki_link.start)

    def test_end_property(self):
        wiki_link = WikiLink("WikiTitle", "link text", 0, 3)
        self.assertEqual(3, wiki_link.end)

    def test_span_property(self):
        wiki_link = WikiLink("WikiTitle", "link text", 0, 3)
        self.assertEqual((0, 3), wiki_link.span)


db = None
db_file = None


class TestDumpDB(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global db, db_file
        dump_file = pkg_resources.resource_filename("tests", "test_data/enwiki-pages-articles-sample.xml.bz2")
        dump_reader = WikiDumpReader(dump_file)
        db_file = NamedTemporaryFile()

        DumpDB.build(dump_reader, db_file.name, 1, 1)
        db = DumpDB(db_file.name)

    @classmethod
    def tearDownClass(cls):
        db_file.close()

    def test_uuid_property(self):
        self.assertIsInstance(db.uuid, str)
        self.assertEqual(32, len(db.uuid))

    def test_dump_file_property(self):
        self.assertTrue(db.dump_file.endswith("enwiki-pages-articles-sample.xml.bz2"))

    def test_language_property(self):
        self.assertEqual("en", db.language)

    def test_page_size(self):
        self.assertEqual(2, db.page_size())

    def test_titles_generator(self):
        self.assertEqual(["Accessibility", "Computer accessibility"], list(db.titles()))

    def test_redirects_generator(self):
        self.assertEqual([("AccessibleComputing", "Computer accessibility")], list(db.redirects()))

    def test_resolve_redirect(self):
        self.assertEqual("Computer accessibility", db.resolve_redirect("AccessibleComputing"))

    def test_is_redirect(self):
        self.assertTrue(db.is_redirect("AccessibleComputing"))

    def test_is_disambiguation(self):
        self.assertFalse(db.is_disambiguation("Computer accessibility"))

    def test_get_paragraphs(self):
        paragraphs = db.get_paragraphs("Computer accessibility")
        paragraph = paragraphs[0]

        self.assertTrue(paragraph.text.lstrip().startswith("In human–computer interaction"))
        self.assertTrue(paragraph.abstract)
        wiki_link = paragraph.wiki_links[0]
        self.assertEqual("Human–computer interaction", wiki_link.title)
        self.assertEqual("human–computer interaction", wiki_link.text)
        self.assertEqual((5, 31), wiki_link.span)

        for paragraph in paragraphs:
            self.assertIsInstance(paragraph, Paragraph)
            for wiki_link in paragraph.wiki_links:
                self.assertIsInstance(wiki_link, WikiLink)
                self.assertEqual(paragraph.text[wiki_link.start : wiki_link.end], wiki_link.text)

    def test_get_paragraphs_with_invalid_key(self):
        self.assertRaises(KeyError, db.get_paragraphs, "foo")

    def test_parse_page(self):
        page = WikiPage(
            title="Japan",
            language="en",
            wiki_text="'''Japan''' is a [[Sovereign state|sovereign]] [[island country|island nation]] in <b>[[East Asia]]</b>",
            redirect=None,
        )
        ret = dump_db._parse_page(page, None)
        self.assertEqual("page", ret[0])
        self.assertEqual(b"Japan", ret[1][0])
        paragraph = pickle.loads(zlib.decompress(ret[1][1]))[0][0]
        self.assertEqual("Japan is a sovereign island nation in East Asia", paragraph[0])
        self.assertEqual(
            [
                ("Sovereign state", "sovereign", 11, 20),
                ("Island country", "island nation", 21, 34),
                ("East Asia", "East Asia", 38, 47),
            ],
            paragraph[1],
        )

    def test_parse_page_with_category_link(self):
        page = WikiPage(
            title="Japan",
            language="en",
            wiki_text="[[Category:Asia]]",
            redirect=None,
        )
        ret = dump_db._parse_page(page, None)
        paragraph = pickle.loads(zlib.decompress(ret[1][1]))[0][0]
        # Category prefix should be removed
        self.assertEqual("Asia", paragraph[0])
        self.assertEqual([("Category:Asia", "Asia", 0, 4)], paragraph[1])

        page = WikiPage(
            title="Japan",
            language="en",
            wiki_text="[[Category:Asia|Asian countries]]",
            redirect=None,
        )
        ret = dump_db._parse_page(page, None)
        paragraph = pickle.loads(zlib.decompress(ret[1][1]))[0][0]
        self.assertEqual("Asian countries", paragraph[0])
        self.assertEqual([("Category:Asia", "Asian countries", 0, 15)], paragraph[1])

    def test_parse_page_with_media_link(self):
        page = WikiPage(
            title="Japan",
            language="en",
            wiki_text="[[File:japan.csv]] / [[Image:japan.png|Image]]",
            redirect=None,
        )
        ret = dump_db._parse_page(page, None)
        paragraph = pickle.loads(zlib.decompress(ret[1][1]))[0][0]
        # Media links should be removed
        self.assertEqual(" / ", paragraph[0])
        self.assertFalse(paragraph[1])

    def test_parse_page_redirect(self):
        page = WikiPage(title="日本", language="en", wiki_text="#REDIRECT [[Japan]]", redirect="Japan")
        ret = dump_db._parse_page(page, None)
        self.assertEqual("redirect", ret[0])
        self.assertEqual("日本".encode("utf-8"), ret[1][0])
        self.assertEqual(b"Japan", ret[1][1])


if __name__ == "__main__":
    unittest.main()
