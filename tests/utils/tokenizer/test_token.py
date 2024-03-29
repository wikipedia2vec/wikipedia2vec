import unittest

from wikipedia2vec.utils.tokenizer.token import Token


class TestToken(unittest.TestCase):
    def test_text_property(self):
        token = Token("text", 1, 3)
        self.assertEqual("text", token.text)

    def test_start_property(self):
        token = Token("text", 1, 3)
        self.assertEqual(1, token.start)

    def test_end_property(self):
        token = Token("text", 1, 3)
        self.assertEqual(3, token.end)

    def test_span_property(self):
        token = Token("text", 1, 3)
        self.assertEqual((1, 3), token.span)


if __name__ == "__main__":
    unittest.main()
