import unittest

from wikipedia2vec.utils.tokenizer.token import Token

try:
    import MeCab

    MECAB_INSTALLED = True
except ImportError:
    MECAB_INSTALLED = False


if MECAB_INSTALLED:

    class TestMeCabTokenizer(unittest.TestCase):
        def setUp(self):
            from wikipedia2vec.utils.tokenizer.mecab_tokenizer import MeCabTokenizer

            self._tokenizer = MeCabTokenizer()

        def test_tokenize(self):
            text = "東京は日本の首都です"
            tokens = self._tokenizer.tokenize(text)

            for token in tokens:
                self.assertIsInstance(token, Token)
            self.assertEqual(["東京", "は", "日本", "の", "首都", "です"], [t.text for t in tokens])
            self.assertEqual([(0, 2), (2, 3), (3, 5), (5, 6), (6, 8), (8, 10)], [t.span for t in tokens])


if __name__ == "__main__":
    unittest.main()
