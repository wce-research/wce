from dataclasses import dataclass
import re

from language.lexer import normalize
from language.parser import parse_action
from language.query import (
    QueryParseError,
    parse_query,
)


class CommandParseError(ValueError):
    pass


@dataclass(frozen=True)
class ActionCommand:
    action: object


@dataclass(frozen=True)
class QueryCommand:
    query: object


@dataclass(frozen=True)
class AdvanceTimeCommand:
    duration: float


def parse_command(text: str):
    normalized = normalize(text)

    # Commande temporelle :
    # "Avance le temps de 2 secondes"
    match = re.match(
        r"^avance\s+le\s+temps\s+de\s+"
        r"(?P<duration>\d+(?:\.\d+)?)\s*"
        r"(?:s|seconde|secondes)?\.?$",
        normalized,
    )

    if match:
        return AdvanceTimeCommand(
            duration=float(
                match.group("duration")
            )
        )

    # On ne devine plus si c'est une question
    # à partir du premier mot.
    # On laisse le parseur de questions décider.
    try:
        query = parse_query(text)

        return QueryCommand(
            query=query,
        )

    except QueryParseError:
        pass

    # Sinon, essayer comme action.
    try:
        action = parse_action(text)

        return ActionCommand(
            action=action,
        )

    except ValueError as error:
        raise CommandParseError(
            str(error)
        ) from error
