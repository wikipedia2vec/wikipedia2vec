from typing import Tuple

import cython


@cython.cclass
class Sentence:
    text = cython.declare(str, visibility="readonly")
    start = cython.declare(cython.int, visibility="readonly")
    end = cython.declare(cython.int, visibility="readonly")

    def __init__(self, text: str, start: int, end: int):
        self.text = text
        self.start = start
        self.end = end

    @property
    def span(self) -> Tuple[int, int]:
        return (self.start, self.end)

    def __reduce__(self):
        return (self.__class__, (self.text, self.start, self.end))

    def __repr__(self):
        return f"<Sentence {self.text}>"
