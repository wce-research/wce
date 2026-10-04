from core.executor import execute_action
from language.parser import parse_action


def execute_text(world, text: str):
    action = parse_action(text)

    return execute_action(
        world,
        action,
    )
