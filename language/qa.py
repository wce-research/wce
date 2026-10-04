from language.answer import answer_query
from language.query import parse_query


def ask(world, text: str) -> str:
    query = parse_query(text)

    return answer_query(
        world,
        query,
    )
