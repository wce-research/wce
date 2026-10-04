from dataclasses import dataclass

from language.lexer import strip_accents, tokenize


class QueryParseError(ValueError):
    pass


@dataclass(frozen=True)
class Query:
    kind: str
    entity: str | None = None


ARTICLES = {
    "le",
    "la",
    "les",
    "un",
    "une",
    "l",
    "du",
    "de",
}


def canonical(token: str) -> str:
    return (
        strip_accents(token)
        .lower()
        .replace("'", "")
    )


def parse_query(text: str) -> Query:
    tokens = tokenize(text)

    if not tokens:
        raise QueryParseError("Question vide")

    words = [
        canonical(token)
        for token in tokens
    ]

    # "Quel est le temps ?"
    if words[:4] == [
        "quel",
        "est",
        "le",
        "temps",
    ]:
        return Query(
            kind="time",
        )

    # "Combien de temps s'est écoulé ?"
    if (
        len(words) >= 3
        and words[0] == "combien"
        and words[1] == "de"
        and words[2] == "temps"
    ):
        return Query(
            kind="time",
        )

    # "Où est la balle ?"
    if (
        len(words) >= 3
        and words[0] == "ou"
        and words[1] == "est"
    ):
        entity = _last_entity(words[2:])

        return Query(
            kind="location",
            entity=entity,
        )

    # "Quelle est la hauteur de la balle ?"
    if (
        len(words) >= 5
        and words[0] in {"quelle", "quel"}
        and words[1] == "est"
        and words[2] == "la"
        and words[3] == "hauteur"
    ):
        entity = _last_entity(words[4:])

        return Query(
            kind="height",
            entity=entity,
        )

    # "Quelle est la position horizontale de la balle ?"
    if (
        len(words) >= 6
        and words[0] in {"quelle", "quel"}
        and words[1] == "est"
        and words[2] == "la"
        and words[3] == "position"
        and words[4] == "horizontale"
    ):
        entity = _last_entity(words[5:])

        return Query(
            kind="x_position",
            entity=entity,
        )

    # "Quelle est la position de la balle ?"
    if (
        len(words) >= 5
        and words[0] in {"quelle", "quel"}
        and words[1] == "est"
        and words[2] == "la"
        and words[3] == "position"
    ):
        entity = _last_entity(words[4:])

        return Query(
            kind="position",
            entity=entity,
        )

    # "Où est la balle en x ?"
    if (
        len(words) >= 5
        and words[0] == "ou"
        and words[1] == "est"
        and words[3] == "en"
        and words[4] == "x"
    ):
        entity = _last_entity(words[2:3])

        return Query(
            kind="x_position",
            entity=entity,
        )

    # "Quelle est la vitesse horizontale de la balle ?"
    if (
        len(words) >= 6
        and words[0] in {"quelle", "quel"}
        and words[1] == "est"
        and words[2] == "la"
        and words[3] == "vitesse"
        and words[4] == "horizontale"
    ):
        entity = _last_entity(words[5:])

        return Query(
            kind="vx",
            entity=entity,
        )

    # "Quelle est la vitesse verticale de la balle ?"
    if (
        len(words) >= 6
        and words[0] in {"quelle", "quel"}
        and words[1] == "est"
        and words[2] == "la"
        and words[3] == "vitesse"
        and words[4] == "verticale"
    ):
        entity = _last_entity(words[5:])

        return Query(
            kind="vy",
            entity=entity,
        )

    # "Quelle est la vitesse de la balle ?"
    if (
        len(words) >= 5
        and words[0] in {"quelle", "quel"}
        and words[1] == "est"
        and words[2] == "la"
        and words[3] == "vitesse"
    ):
        entity = _last_entity(words[4:])

        return Query(
            kind="velocity",
            entity=entity,
        )

    raise QueryParseError(
        f"Question non comprise : {text!r}"
    )


def _last_entity(words: list[str]) -> str:
    candidates = [
        word
        for word in words
        if word not in ARTICLES
        and word != "de"
    ]

    if not candidates:
        raise QueryParseError(
            "Entité manquante"
        )

    return candidates[-1]
