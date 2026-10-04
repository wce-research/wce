from core.executor import execute_action
from core.world import Entity, World

from language.answer import answer_query
from language.command import (
    ActionCommand,
    AdvanceTimeCommand,
    QueryCommand,
    parse_command,
)

from physics.constraints import non_negative_height
from physics.engine import simulate
from physics.interactions import GroundCollision
from physics.laws import GravityLaw


class Session:
    def __init__(self):
        self.world = self.build_world()

        self.laws = [
            GravityLaw(),
        ]

        self.interactions = [
            GroundCollision(
                ground_y=0.0,
                restitution=0.0,
                support_name="sol",
            ),
        ]

        self.dt = 0.1

    def build_world(self):
        world = World()

        world.add_entity(
            Entity(
                name="sol",
                kind="support",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "fixed": True,
                },
            )
        )

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_relation(
            "balle",
            "supported_by",
            "sol",
        )

        world.add_constraint(
            non_negative_height()
        )

        return world

    def execute(self, text: str) -> str:
        command = parse_command(text)

        if isinstance(
            command,
            ActionCommand,
        ):
            execute_action(
                self.world,
                command.action,
            )

            return "OK."

        if isinstance(
            command,
            QueryCommand,
        ):
            return answer_query(
                self.world,
                command.query,
            )

        if isinstance(
            command,
            AdvanceTimeCommand,
        ):
            simulate(
                self.world,
                laws=self.laws,
                interactions=self.interactions,
                duration=command.duration,
                dt=self.dt,
            )

            return "OK."

        raise RuntimeError(
            f"Commande inconnue : {command!r}"
        )
