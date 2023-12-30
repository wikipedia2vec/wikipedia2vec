import pkg_resources
import unittest

from wikipedia2vec.utils.utils import is_category_title, normalize_title


class TestUtils(unittest.TestCase):
    def test_is_category_title(self):
        self.assertTrue(is_category_title("Category:PageTitle", "en"))
        self.assertFalse(is_category_title("PageTitle", "en"))
        self.assertTrue(is_category_title("カテゴリ:PageTitle", "ja"))
        self.assertFalse(is_category_title("カテゴリ:PageTitle", "en"))

    def test_normalize_title(self):
        self.assertEqual("Yokohama station", normalize_title("yokohama_station"))


if __name__ == "__main__":
    unittest.main()
