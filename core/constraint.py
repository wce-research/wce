from dataclasses import dataclass
from typing import Callable


@dataclass
class Constraint:
    name: str
    check: Callable

    def validate(self, world) -> tuple[bool, str]:
        return self.check(world)
