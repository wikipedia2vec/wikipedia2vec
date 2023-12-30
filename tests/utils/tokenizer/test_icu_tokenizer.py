import unittest

from wikipedia2vec.utils.tokenizer.token import Token
from wikipedia2vec.utils.tokenizer.icu_tokenizer import ICUTokenizer


class TestICUTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = ICUTokenizer("en")

    def test_tokenize(self):
        text = "Tokyo is the capital of Japan"
        tokens = self._tokenizer.tokenize(text)

        for token in tokens:
            self.assertIsInstance(token, Token)
        self.assertEqual(["Tokyo", "is", "the", "capital", "of", "Japan"], [t.text for t in tokens])
        self.assertEqual([(0, 5), (6, 8), (9, 12), (13, 20), (21, 23), (24, 29)], [t.span for t in tokens])


if __name__ == "__main__":
    unittest.main()
