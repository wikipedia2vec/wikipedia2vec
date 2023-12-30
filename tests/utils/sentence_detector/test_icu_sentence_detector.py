import unittest

from wikipedia2vec.utils.sentence_detector.sentence import Sentence
from wikipedia2vec.utils.sentence_detector.icu_sentence_detector import ICUSentenceDetector


class TestICUSentenceDetector(unittest.TestCase):
    def setUp(self):
        self._detector = ICUSentenceDetector("en")

    def test_detect_sentences(self):
        text = "Wikipedia is an encyclopedia based on a model of openly editable content. It is the largest general reference work on the Internet."
        sents = self._detector.detect_sentences(text)

        for sent in sents:
            self.assertIsInstance(sent, Sentence)
        self.assertEqual("Wikipedia is an encyclopedia based on a model of openly editable content. ", sents[0].text)
        self.assertEqual("It is the largest general reference work on the Internet.", sents[1].text)
        self.assertEqual([(0, 74), (74, 131)], [s.span for s in sents])


if __name__ == "__main__":
    unittest.main()
