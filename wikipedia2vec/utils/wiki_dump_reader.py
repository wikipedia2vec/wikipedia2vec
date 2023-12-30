import bz2
import logging
import re
from typing import Iterator, Tuple

from xml.etree.ElementTree import iterparse

from . import normalize_title
from .wiki_page import WikiPage

logger = logging.getLogger(__name__)

NAMESPACE_RE = re.compile(r"^{(.*?)}")


class WikiDumpReader:
    def __init__(self, dump_file: str):
        self._dump_file = dump_file
        with bz2.open(self._dump_file, "rt") as f:
            self._language = re.search(r'xml:lang="(.*)"', f.readline()).group(1)

    @property
    def dump_file(self) -> str:
        return self._dump_file

    @property
    def language(self) -> str:
        return self._language

    def __iter__(self) -> Iterator[WikiPage]:
        c = 0
        for title, wiki_text, redirect in _extract_pages(self._dump_file):
            c += 1

            yield WikiPage(title, self._language, wiki_text, redirect)

            if c % 100000 == 0:
                logger.info(f"Processed: {c} pages")


def _extract_pages(dump_file: str) -> Iterator[Tuple[str, str, str]]:
    with bz2.open(dump_file, "rt") as f:
        context = iterparse(f, events=("end",))
        for _, elem in context:
            if not elem.tag.endswith("page"):
                continue
            namespace = elem.tag[:-4]

            title = elem.find(f"./{namespace}title").text
            ns = elem.find(f"./{namespace}ns").text
            redirect = elem.find(f"./{namespace}redirect")
            if redirect is not None:
                redirect = normalize_title(redirect.attrib["title"])

            # filter pages that are not in the "main" namespace.
            if ns != "0":
                elem.clear()
                continue

            text = elem.find(f"./{namespace}revision/{namespace}text").text or ""
            elem.clear()

            yield title, text, redirect
