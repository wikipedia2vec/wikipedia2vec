from typing import List, Tuple

import MeCab

from .base_tokenizer import BaseTokenizer


class MeCabTokenizer(BaseTokenizer):
    def __init__(self):
        self._tagger = MeCab.Tagger("")

    def _span_tokenize(self, text: str) -> List[Tuple[int, int]]:
        node = self._tagger.parseToNode(text)

        cur = 0
        ret = []

        while node:
            if node.stat not in (2, 3):
                word = node.surface
                space_length = node.rlength - node.length

                start = cur + space_length
                end = start + len(word)

                ret.append((start, end))

                cur += len(word) + space_length

            node = node.next

        return ret

    def __reduce__(self):
        return (self.__class__, tuple())
