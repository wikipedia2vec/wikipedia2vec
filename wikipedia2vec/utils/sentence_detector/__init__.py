from typing import Optional


def get_sentence_detector(name: str, language: Optional[str] = None):
    if name == "icu":
        from .icu_sentence_detector import ICUSentenceDetector

        return ICUSentenceDetector(language)
    else:
        raise NotImplementedError()
