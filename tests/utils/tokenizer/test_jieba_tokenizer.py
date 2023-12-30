import unittest

from wikipedia2vec.utils.tokenizer.token import Token
from wikipedia2vec.utils.tokenizer.jieba_tokenizer import JiebaTokenizer


class TestJiebaTokenizer(unittest.TestCase):
    def setUp(self):
        self._tokenizer = JiebaTokenizer()

    def test_tokenize(self):
        text = "中華人民共和國的首都是北京。"
        tokens = self._tokenizer.tokenize(text)

        for token in tokens:
            self.assertIsInstance(token, Token)
        self.assertEqual(["中華", "人民", "共和", "國的", "首都", "是", "北京", "。"], [t.text for t in tokens])
        self.assertEqual(
            [(0, 2), (2, 4), (4, 6), (6, 8), (8, 10), (10, 11), (11, 13), (13, 14)], [t.span for t in tokens]
        )


if __name__ == "__main__":
    unittest.main()
