import re
import unicodedata


def strip_accents(text: str) -> str:
    return "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )


def normalize(text: str) -> str:
    text = text.strip().lower()
    text = text.replace(",", ".")
    return text


def tokenize(text: str) -> list[str]:
    text = normalize(text)

    return re.findall(
        r"\d+(?:\.\d+)?|[\w'-]+",
        text,
        flags=re.UNICODE,
    )
