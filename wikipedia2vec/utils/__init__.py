def normalize_title(title: str) -> str:
    return (title[0].upper() + title[1:]).replace("_", " ")
