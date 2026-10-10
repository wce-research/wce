import unittest

from app.session import Session
from core.action import (
    ImpulseAction,
    LiftAction,
    PushAction,
)
from core.executor import execute_action
from core.quantity import make_quantity
from core.world import Entity, World

from language.qa import ask
from language.query import parse_query

from language.parser import (
    ParseError,
    extract_intent,
    parse_action,
)

from physics.constraints import non_negative_height
from physics.engine import simulate
from physics.interactions import (
    GroundCollision,
    WallCollision,
)
from physics.forces import (
    ConstantForce,
    ExternalForce,
    GravityForce,
    SupportForce,
    net_force,
    net_force_y,
)   
from physics.laws import (
    ForceDynamicsLaw,
    GravityLaw,
)


def build_supported_ball():
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


class QuantityTests(unittest.TestCase):

    def test_centimeters_to_meters(self):
        quantity = make_quantity(250, "cm")

        self.assertEqual(
            quantity.dimension,
            "length",
        )

        self.assertAlmostEqual(
            quantity.value,
            2.5,
        )

        self.assertEqual(
            quantity.unit,
            "m",
        )


class LanguageTests(unittest.TestCase):

    def test_equivalent_formulations(self):
        cases = {
            "Soulève la balle à 2 mètres": 2.0,
            "Monte la balle à 200 cm": 2.0,
            "Lève une balle à 2000 mm": 2.0,
        }

        for text, expected in cases.items():
            with self.subTest(text=text):
                action = parse_action(text)

                self.assertIsInstance(
                    action,
                    LiftAction,
                )

                self.assertEqual(
                    action.entity_name,
                    "balle",
                )

                self.assertAlmostEqual(
                    action.target_y,
                    expected,
                )

    def test_unknown_verb_is_rejected(self):
        with self.assertRaises(ParseError):
            parse_action(
                "Fais danser la balle"
            )

    def test_intent_is_canonical(self):
        intent = extract_intent(
            "Monte la balle à 300 cm"
        )

        self.assertEqual(
            intent.action,
            "lift",
        )

        self.assertEqual(
            intent.entity,
            "balle",
        )

        self.assertEqual(
            intent.quantity.dimension,
            "length",
        )

        self.assertAlmostEqual(
            intent.quantity.value,
            3.0,
        )


class QueryTests(unittest.TestCase):

    def test_parse_location_query(self):
        query = parse_query(
            "Où est la balle ?"
        )

        self.assertEqual(
            query.kind,
            "location",
        )

        self.assertEqual(
            query.entity,
            "balle",
        )

    def test_answer_supported_location(self):
        world = build_supported_ball()

        answer = ask(
            world,
            "Où est la balle ?",
        )

        self.assertEqual(
            answer,
            "Balle est sur sol.",
        )

    def test_answer_height(self):
        world = build_supported_ball()

        execute_action(
            world,
            LiftAction(
                entity_name="balle",
                target_y=2.5,
            ),
        )

        answer = ask(
            world,
            "Quelle est la hauteur de la balle ?",
        )

        self.assertEqual(
            answer,
            "La hauteur de balle est de 2.5 m.",
        )

    def test_location_after_lift(self):
        world = build_supported_ball()

        execute_action(
            world,
            LiftAction(
                entity_name="balle",
                target_y=2.0,
            ),
        )

        answer = ask(
            world,
            "Où est la balle ?",
        )

        self.assertEqual(
            answer,
            "Balle se trouve à 2 m de hauteur.",
        )


class PhysicsTests(unittest.TestCase):

    def test_gravity(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 10.0,
                    "vy": 0.0,
                },
            )
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(
            ball.properties["vy"],
            -9.81,
        )

        self.assertAlmostEqual(
            ball.properties["y"],
            4.6045,
        )

    def test_collision_creates_support(self):
        world = build_supported_ball()

        execute_action(
            world,
            LiftAction(
                entity_name="balle",
                target_y=2.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            interactions=[
                GroundCollision(
                    ground_y=0.0,
                    restitution=0.0,
                    support_name="sol",
                )
            ],
            duration=2.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(
            ball.properties["y"],
            0.0,
        )

        self.assertAlmostEqual(
            ball.properties["vy"],
            0.0,
        )

        relations = [
            (
                relation.subject,
                relation.predicate,
                relation.object,
            )
            for relation in world.relations
        ]

        self.assertIn(
            (
                "balle",
                "supported_by",
                "sol",
            ),
            relations,
        )


class ActionTests(unittest.TestCase):

    def test_lift_removes_support(self):
        world = build_supported_ball()

        execute_action(
            world,
            LiftAction(
                entity_name="balle",
                target_y=2.0,
            ),
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(
            ball.properties["y"],
            2.0,
        )

        self.assertFalse(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                for relation in world.relations
            )
        )


class SessionTests(unittest.TestCase):

    def test_initial_location(self):
        session = Session()

        self.assertEqual(
            session.execute(
                "Où est la balle ?"
            ),
            "Balle est sur sol.",
        )

    def test_lift_then_query(self):
        session = Session()

        session.execute(
            "Soulève la balle à 2 mètres"
        )

        self.assertEqual(
            session.execute(
                "Où est la balle ?"
            ),
            "Balle se trouve à 2 m de hauteur.",
        )

    def test_time_causes_fall(self):
        session = Session()

        session.execute(
            "Soulève la balle à 2 mètres"
        )

        session.execute(
            "Avance le temps de 2 secondes"
        )

        self.assertEqual(
            session.execute(
                "Où est la balle ?"
            ),
            "Balle est sur sol.",
        )

    def test_session_preserves_world(self):
        session = Session()

        session.execute(
            "Monte la balle à 250 cm"
        )

        self.assertEqual(
            session.execute(
                "Quelle est la hauteur de la balle ?"
            ),
            "La hauteur de balle est de 2.5 m.",
        )


class TimeTests(unittest.TestCase):

    def test_world_starts_at_zero(self):
        world = World()

        self.assertEqual(
            world.time,
            0.0,
        )

    def test_world_time_advances(self):
        session = Session()

        session.execute(
            "Avance le temps de 1 seconde"
        )

        self.assertAlmostEqual(
            session.world.time,
            1.0,
        )

    def test_fractional_time_is_exact(self):
        session = Session()

        session.execute(
            "Avance le temps de 0.35 secondes"
        )

        self.assertAlmostEqual(
            session.world.time,
            0.35,
        )

    def test_time_query(self):
        session = Session()

        session.execute(
            "Avance le temps de 2 secondes"
        )

        self.assertEqual(
            session.execute(
                "Quel est le temps ?"
            ),
            "Le temps du monde est de 2 s.",
        )

    def test_elapsed_time_query(self):
        session = Session()

        session.execute(
            "Avance le temps de 1 seconde"
        )

        self.assertEqual(
            session.execute(
                "Combien de temps s'est écoulé ?"
            ),
            "Le temps du monde est de 1 s.",
        )


class CommandRoutingTests(unittest.TestCase):

    def test_time_query_is_routed_as_query(self):
        session = Session()

        self.assertEqual(
            session.execute(
                "Quel est le temps ?"
            ),
            "Le temps du monde est de 0 s.",
        )

    def test_time_query_with_leading_unicode_character(self):
        session = Session()

        self.assertEqual(
            session.execute(
                "\ufeffQuel est le temps ?"
            ),
            "Le temps du monde est de 0 s.",
        )


class ForceTests(unittest.TestCase):

    def test_multiple_forces_are_summed(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 0.0,
                    "vy": 0.0,
                    "masse": 2.0,
                },
            )
        )

        providers = [
            ConstantForce(
                "balle",
                10.0,
                source="force_a",
            ),
            ConstantForce(
                "balle",
                -4.0,
                source="force_b",
            ),
        ]

        self.assertAlmostEqual(
            net_force_y(
                world,
                providers,
                "balle",
            ),
            6.0,
        )

    def test_zero_net_force_means_no_acceleration(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    5.0,
                ),
                ConstantForce(
                    "balle",
                    -5.0,
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(
            ball.properties["vy"],
            0.0,
        )

        self.assertAlmostEqual(
            ball.properties["y"],
            5.0,
        )

    def test_support_balances_gravity(self):
        world = build_supported_ball()

        providers = [
            GravityForce(),
            SupportForce(),
        ]

        self.assertAlmostEqual(
            net_force_y(
                world,
                providers,
                "balle",
            ),
            0.0,
        )

    def test_unbalanced_force_accelerates_object(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 0.0,
                    "vy": 0.0,
                    "masse": 2.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    10.0,
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # F = 10 N
        # m = 2 kg
        # a = 5 m/s²
        # after 1 second:
        # vy = 5 m/s
        self.assertAlmostEqual(
            ball.properties["vy"],
            5.0,
        )

class PushTests(unittest.TestCase):

    def test_newton_quantity(self):
        quantity = make_quantity(
            15,
            "n",
        )

        self.assertEqual(
            quantity.dimension,
            "force",
        )

        self.assertAlmostEqual(
            quantity.value,
            15.0,
        )

        self.assertEqual(
            quantity.unit,
            "N",
        )

    def test_push_action_sets_external_force(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 0.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=15.0,
            ),
        )

        self.assertAlmostEqual(
            world.entities["balle"].properties[
                "external_force_y"
            ],
            15.0,
        )

    def test_language_parses_push(self):
        action = parse_action(
            "Pousse la balle à 15 N"
        )

        self.assertIsInstance(
            action,
            PushAction,
        )

        self.assertEqual(
            action.entity_name,
            "balle",
        )

        self.assertAlmostEqual(
            action.force_y,
            15.0,
        )

    def test_push_changes_motion(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 0.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=15.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # Net force:
        # 15 - 9.81 = 5.19 N
        #
        # m = 1 kg
        #
        # a = 5.19 m/s²
        #
        # after 1 second:
        # vy = 5.19 m/s
        self.assertAlmostEqual(
            ball.properties["vy"],
            5.19,
            places=6,
        )

    def test_weak_push_does_not_lift_supported_ball(self):
        world = build_supported_ball()

        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=5.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(
            ball.properties["y"],
            0.0,
        )

        self.assertAlmostEqual(
            ball.properties["vy"],
            0.0,
        )

        self.assertTrue(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                and relation.object == "sol"
                for relation in world.relations
            )
        )

    def test_strong_push_breaks_support(self):
        world = build_supported_ball()

        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=15.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertFalse(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                for relation in world.relations
            )
        )

        # Net force:
        # 15 - 9.81 = 5.19 N
        #
        # m = 1 kg
        # a = 5.19 m/s²
        #
        # after 0.1 s:
        # vy = 0.519 m/s
        self.assertAlmostEqual(
            ball.properties["vy"],
            0.519,
        )

        # Semi-implicit Euler:
        # y = 0 + 0.519 * 0.1
        self.assertAlmostEqual(
            ball.properties["y"],
            0.0519,
        )

class ImpulseTests(unittest.TestCase):

    def test_impulse_quantity(self):
        quantity = make_quantity(
            3,
            "ns",
        )

        self.assertEqual(
            quantity.dimension,
            "impulse",
        )

        self.assertAlmostEqual(
            quantity.value,
            3.0,
        )

        self.assertEqual(
            quantity.unit,
            "N·s",
        )

    def test_impulse_changes_velocity(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 0.0,
                    "vy": 0.0,
                    "masse": 2.0,
                },
            )
        )

        execute_action(
            world,
            ImpulseAction(
                entity_name="balle",
                impulse_y=4.0,
            ),
        )

        ball = world.entities["balle"]

        # Δv = J / m = 4 / 2 = 2 m/s
        self.assertAlmostEqual(
            ball.properties["vy"],
            2.0,
        )

    def test_impulse_is_not_persistent_force(self):
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "y": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        execute_action(
            world,
            ImpulseAction(
                entity_name="balle",
                impulse_y=3.0,
            ),
        )

        self.assertNotIn(
            "external_force_y",
            world.entities["balle"].properties,
        )

    def test_language_parses_impulse(self):
        action = parse_action(
            "Impulse la balle à 3 Ns"
        )

        self.assertIsInstance(
            action,
            ImpulseAction,
        )

        self.assertEqual(
            action.entity_name,
            "balle",
        )

        self.assertAlmostEqual(
            action.impulse_y,
            3.0,
        )
    
    def test_upward_impulse_removes_support(self):
        world = build_supported_ball()

        execute_action(
            world,
            ImpulseAction(
                entity_name="balle",
                impulse_y=3.0,
            ),
        )

        self.assertFalse(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                for relation in world.relations
            )
        )

        self.assertAlmostEqual(
            world.entities["balle"].properties["vy"],
            3.0,
        )

    def test_impulse_then_gravity(self):
        world = build_supported_ball()

        execute_action(
            world,
            ImpulseAction(
                entity_name="balle",
                impulse_y=5.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            interactions=[
                GroundCollision(
                    ground_y=0.0,
                    restitution=0.0,
                    support_name="sol",
                )
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertGreater(
            ball.properties["y"],
            0.0,
        )

        self.assertAlmostEqual(
            ball.properties["vy"],
            5.0 - 0.981,
        )


class Physics2DTests(unittest.TestCase):
    """Tests for 2D physics."""

    def test_horizontal_force_changes_vx_and_x(self):
        """A horizontal force alone modifies vx and x."""
        world = World()

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

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=10.0,
                    fy=0.0,
                    source="horizontal_push",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # Fx = 10 N
        # m = 1 kg
        # ax = 10 m/s²
        # after 1 second:
        # vx = 10 m/s
        self.assertAlmostEqual(
            ball.properties["vx"],
            10.0,
        )

        # Semi-implicit Euler: x(t+dt) = x(t) + v(t+dt)*dt
        # After 10 steps of 0.1s: sum of (i*10*0.1 * 0.1) for i=1..10
        # = 0.1 * (0.1 + 0.2 + ... + 1.0) * 10
        # = 0.1 * 5.5 * 10 = 5.5
        # But in fact: vx after each step = 1, 2, 3, ..., 10
        # x = 0 + 1*0.1 + 2*0.1 + 3*0.1 + ... + 10*0.1
        #   = 0.1 * (1+2+3+...+10) = 0.1 * 55 = 5.5
        self.assertAlmostEqual(
            ball.properties["x"],
            5.5,
        )

    def test_vertical_force_unchanged_behavior(self):
        """A vertical force alone preserves the existing 1D behaviour."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "masse": 2.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fy=10.0,
                    source="vertical_push",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # Fy = 10 N
        # m = 2 kg
        # ay = 5 m/s²
        # after 1 second:
        # vy = 5 m/s
        self.assertAlmostEqual(
            ball.properties["vy"],
            5.0,
        )

        # Semi-implicit Euler: y = sum of (vy_i * dt)
        # vy after each step = 0.5, 1.0, 1.5, ..., 5.0
        # y = 0.1 * (0.5 + 1.0 + 1.5 + ... + 5.0)
        #    = 0.1 * 27.5 = 2.75
        self.assertAlmostEqual(
            ball.properties["y"],
            2.75,
        )

        # vx and x do not change
        self.assertAlmostEqual(
            ball.properties["vx"],
            0.0,
        )

        self.assertAlmostEqual(
            ball.properties["x"],
            0.0,
        )

    def test_multiple_2d_forces_sum_correctly(self):
        """Several 2D forces sum correctly."""
        world = World()

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

        providers = [
            ConstantForce("balle", fx=5.0, fy=3.0, source="force_a"),
            ConstantForce("balle", fx=-2.0, fy=4.0, source="force_b"),
        ]

        fx_net, fy_net = net_force(world, providers, "balle")

        self.assertAlmostEqual(fx_net, 3.0)
        self.assertAlmostEqual(fy_net, 7.0)

    def test_gravity_plus_horizontal_force_combined_trajectory(self):
        """Gravity + horizontal force give a combined trajectory."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 10.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                GravityForce(),
                ConstantForce(
                    "balle",
                    fx=5.0,
                    fy=0.0,
                    source="horizontal",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # Fx = 5 N, m = 1 kg, ax = 5 m/s²
        self.assertAlmostEqual(
            ball.properties["vx"],
            5.0,
        )

        # Fy = -9.81 N, m = 1 kg, ay = -9.81 m/s²
        self.assertAlmostEqual(
            ball.properties["vy"],
            -9.81,
            places=6,
        )

        # Semi-implicit Euler: x = 0.1 * (0.5 + 1.0 + 1.5 + ... + 5.0)
        # = 0.1 * 27.5 = 2.75
        self.assertAlmostEqual(
            ball.properties["x"],
            2.75,
        )

    def test_horizontal_impulse_changes_vx(self):
        """A horizontal impulse changes vx."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "masse": 2.0,
                },
            )
        )

        execute_action(
            world,
            ImpulseAction(
                entity_name="balle",
                impulse_y=0.0,
                impulse_x=4.0,
            ),
        )

        ball = world.entities["balle"]

        # Δvx = Jx / m = 4 / 2 = 2 m/s
        self.assertAlmostEqual(
            ball.properties["vx"],
            2.0,
        )

        # vy does not change
        self.assertAlmostEqual(
            ball.properties["vy"],
            0.0,
        )


class Language2DTests(unittest.TestCase):
    """Tests for 2D natural language."""

    def test_parse_horizontal_push_droite(self):
        """Parse 'Pousse la balle vers la droite à 10 N'."""
        action = parse_action(
            "Pousse la balle vers la droite à 10 N"
        )

        self.assertIsInstance(
            action,
            PushAction,
        )

        self.assertEqual(
            action.entity_name,
            "balle",
        )

        self.assertAlmostEqual(
            action.force_x,
            10.0,
        )

        self.assertAlmostEqual(
            action.force_y,
            0.0,
        )

    def test_parse_horizontal_push_gauche(self):
        """Parse 'Pousse la balle vers la gauche à 5 N'."""
        action = parse_action(
            "Pousse la balle vers la gauche à 5 N"
        )

        self.assertIsInstance(
            action,
            PushAction,
        )

        self.assertEqual(
            action.entity_name,
            "balle",
        )

        self.assertAlmostEqual(
            action.force_x,
            -5.0,
        )

        self.assertAlmostEqual(
            action.force_y,
            0.0,
        )

    def test_parse_horizontal_push_horizontalement(self):
        """Parse 'Pousse la balle horizontalement à 15 N'."""
        action = parse_action(
            "Pousse la balle horizontalement à 15 N"
        )

        self.assertIsInstance(
            action,
            PushAction,
        )

        self.assertEqual(
            action.entity_name,
            "balle",
        )

        self.assertAlmostEqual(
            action.force_x,
            15.0,
        )

        self.assertAlmostEqual(
            action.force_y,
            0.0,
        )

    def test_parse_vertical_push_unchanged(self):
        """Parse 'Pousse la balle à 10 N' stays vertical."""
        action = parse_action(
            "Pousse la balle à 10 N"
        )

        self.assertIsInstance(
            action,
            PushAction,
        )

        self.assertEqual(
            action.entity_name,
            "balle",
        )

        self.assertAlmostEqual(
            action.force_y,
            10.0,
        )

        self.assertAlmostEqual(
            action.force_x,
            0.0,
        )

    def test_x_position_query(self):
        """Query 'Quelle est la position horizontale de la balle ?'."""
        world = build_supported_ball()

        # Move the ball horizontally
        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=0.0,
                force_x=10.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=1.0,
            dt=0.1,
        )

        query = parse_query(
            "Quelle est la position horizontale de la balle ?"
        )

        self.assertEqual(
            query.kind,
            "x_position",
        )

        self.assertEqual(
            query.entity,
            "balle",
        )

        answer = ask(
            world,
            "Quelle est la position horizontale de la balle ?",
        )

        # The ball should have moved horizontally
        self.assertIn(
            "position horizontale",
            answer,
        )

        self.assertIn(
            "balle",
            answer,
        )

    def test_position_2d_query(self):
        """Query 'Quelle est la position de la balle ?'."""
        world = build_supported_ball()

        # Lift the ball so that it is no longer on the support
        execute_action(
            world,
            LiftAction(
                entity_name="balle",
                target_y=2.0,
            ),
        )

        # Move horizontally
        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=0.0,
                force_x=5.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=0.5,
            dt=0.1,
        )

        query = parse_query(
            "Quelle est la position de la balle ?"
        )

        self.assertEqual(
            query.kind,
            "position",
        )

        self.assertEqual(
            query.entity,
            "balle",
        )

        answer = ask(
            world,
            "Quelle est la position de la balle ?",
        )

        # Must contain x= and y=
        self.assertIn(
            "x=",
            answer,
        )

        self.assertIn(
            "y=",
            answer,
        )

        self.assertIn(
            "se trouve en",
            answer,
        )

    def test_position_query_with_supported_entity(self):
        """Position query for an entity on a support."""
        world = build_supported_ball()

        answer = ask(
            world,
            "Quelle est la position de la balle ?",
        )

        self.assertIn(
            "est sur sol",
            answer,
        )


class Action2DTests(unittest.TestCase):
    """Tests for 2D actions via language."""

    def test_horizontal_push_execution(self):
        """Execute a horizontal push via parse_action."""
        world = build_supported_ball()

        action = parse_action(
            "Pousse la balle vers la droite à 20 N"
        )

        execute_action(world, action)

        ball = world.entities["balle"]

        self.assertAlmostEqual(
            ball.properties.get("external_force_x", 0.0),
            20.0,
        )

    def test_horizontal_impulse_from_language(self):
        """Execute a horizontal impulse via language."""
        world = build_supported_ball()

        action = parse_action(
            "Impulse la balle vers la gauche à 6 Ns"
        )

        execute_action(world, action)

        ball = world.entities["balle"]

        self.assertAlmostEqual(
            ball.properties["vx"],
            -6.0,
        )


class Velocity2DTests(unittest.TestCase):
    """Tests for 2D velocity queries."""

    def test_vx_query(self):
        """Query 'Quelle est la vitesse horizontale de la balle ?'."""
        world = build_supported_ball()

        # Apply a horizontal force
        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=0.0,
                force_x=10.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=1.0,
            dt=0.1,
        )

        query = parse_query(
            "Quelle est la vitesse horizontale de la balle ?"
        )

        self.assertEqual(
            query.kind,
            "vx",
        )

        self.assertEqual(
            query.entity,
            "balle",
        )

        answer = ask(
            world,
            "Quelle est la vitesse horizontale de la balle ?",
        )

        self.assertIn(
            "vitesse horizontale",
            answer,
        )

        self.assertIn(
            "m/s",
            answer,
        )

    def test_vy_query(self):
        """Query 'Quelle est la vitesse verticale de la balle ?'."""
        world = build_supported_ball()

        # Lift and drop
        execute_action(
            world,
            LiftAction(
                entity_name="balle",
                target_y=5.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=0.5,
            dt=0.1,
        )

        query = parse_query(
            "Quelle est la vitesse verticale de la balle ?"
        )

        self.assertEqual(
            query.kind,
            "vy",
        )

        answer = ask(
            world,
            "Quelle est la vitesse verticale de la balle ?",
        )

        self.assertIn(
            "vitesse verticale",
            answer,
        )

        self.assertIn(
            "m/s",
            answer,
        )

    def test_velocity_2d_query(self):
        """Query 'Quelle est la vitesse de la balle ?'."""
        world = build_supported_ball()

        # Lift and push horizontally
        execute_action(
            world,
            LiftAction(
                entity_name="balle",
                target_y=5.0,
            ),
        )

        execute_action(
            world,
            PushAction(
                entity_name="balle",
                force_y=0.0,
                force_x=10.0,
            ),
        )

        simulate(
            world,
            laws=[GravityLaw()],
            duration=0.5,
            dt=0.1,
        )

        query = parse_query(
            "Quelle est la vitesse de la balle ?"
        )

        self.assertEqual(
            query.kind,
            "velocity",
        )

        answer = ask(
            world,
            "Quelle est la vitesse de la balle ?",
        )

        self.assertIn(
            "vitesse",
            answer,
        )

        self.assertIn(
            "vx=",
            answer,
        )

        self.assertIn(
            "vy=",
            answer,
        )


class Advanced2DTests(unittest.TestCase):
    """Tests for advanced 2D."""

    def test_horizontal_impulse_preserves_support(self):
        """A purely horizontal impulse does not remove the support."""
        world = build_supported_ball()

        execute_action(
            world,
            ImpulseAction(
                entity_name="balle",
                impulse_y=0.0,
                impulse_x=5.0,
            ),
        )

        # The support must still be present
        self.assertTrue(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                for relation in world.relations
            )
        )

        # vx must have changed
        self.assertAlmostEqual(
            world.entities["balle"].properties["vx"],
            5.0,
        )

        # vy must not have changed
        self.assertAlmostEqual(
            world.entities["balle"].properties["vy"],
            0.0,
        )

    def test_right_impulse_increases_vx(self):
        """An impulse to the right increases vx."""
        world = build_supported_ball()

        action = parse_action(
            "Impulse la balle vers la droite à 10 Ns"
        )

        execute_action(world, action)

        self.assertAlmostEqual(
            world.entities["balle"].properties["vx"],
            10.0,
        )

        self.assertAlmostEqual(
            world.entities["balle"].properties["vy"],
            0.0,
        )

    def test_left_impulse_decreases_vx(self):
        """An impulse to the left decreases vx."""
        world = build_supported_ball()

        action = parse_action(
            "Impulse la balle vers la gauche à 10 Ns"
        )

        execute_action(world, action)

        self.assertAlmostEqual(
            world.entities["balle"].properties["vx"],
            -10.0,
        )

        self.assertAlmostEqual(
            world.entities["balle"].properties["vy"],
            0.0,
        )

    def test_diagonal_force_accelerates_both_axes(self):
        """A diagonal force accelerates along x and y."""
        world = World()

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

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=10.0,
                    fy=20.0,
                    source="diagonal",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # ax = 10/1 = 10 m/s² → vx = 10 m/s
        self.assertAlmostEqual(
            ball.properties["vx"],
            10.0,
        )

        # ay = 20/1 = 20 m/s² → vy = 20 m/s
        self.assertAlmostEqual(
            ball.properties["vy"],
            20.0,
        )

        # x and y must also have moved
        self.assertGreater(
            ball.properties["x"],
            0.0,
        )

        self.assertGreater(
            ball.properties["y"],
            0.0,
        )

    def test_gravity_plus_horizontal_force_independent(self):
        """Gravity + horizontal force: x and y evolve independently."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 10.0,
                    "vx": 0.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                GravityForce(),
                ConstantForce(
                    "balle",
                    fx=5.0,
                    fy=0.0,
                    source="horizontal",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # vx must have increased (Fx=5, m=1 → ax=5)
        self.assertAlmostEqual(
            ball.properties["vx"],
            5.0,
        )

        # vy must have decreased (Fy=-9.81, m=1 → ay=-9.81)
        self.assertAlmostEqual(
            ball.properties["vy"],
            -9.81,
            places=6,
        )

    def test_zero_force_no_acceleration(self):
        """Absence of force on an axis = no artificial acceleration."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 2.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="zero",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # Without force, vx and vy do not change
        self.assertAlmostEqual(
            ball.properties["vx"],
            2.0,
        )

        self.assertAlmostEqual(
            ball.properties["vy"],
            0.0,
        )


class DiagonalVectorTests(unittest.TestCase):
    """Tests for diagonal vectors via natural language."""

    def test_push_right_and_up(self):
        """Push with positive Fx and positive Fy."""
        action = parse_action(
            "Pousse la balle avec 10 N vers la droite et 5 N vers le haut"
        )

        self.assertIsInstance(action, PushAction)
        self.assertAlmostEqual(action.force_x, 10.0)
        self.assertAlmostEqual(action.force_y, 5.0)

    def test_push_left_and_up(self):
        """Push with negative Fx and positive Fy."""
        action = parse_action(
            "Pousse la balle avec 4 N vers la gauche et 3 N vers le haut"
        )

        self.assertIsInstance(action, PushAction)
        self.assertAlmostEqual(action.force_x, -4.0)
        self.assertAlmostEqual(action.force_y, 3.0)

    def test_push_right_and_down(self):
        """Push with positive Fx and negative Fy."""
        action = parse_action(
            "Pousse la balle avec 10 N vers la droite et 5 N vers le bas"
        )

        self.assertIsInstance(action, PushAction)
        self.assertAlmostEqual(action.force_x, 10.0)
        self.assertAlmostEqual(action.force_y, -5.0)

    def test_impulse_right_and_up(self):
        """Impulse with Δvx > 0 and Δvy > 0."""
        action = parse_action(
            "Impulse la balle avec 6 Ns vers la droite et 2 Ns vers le haut"
        )

        self.assertIsInstance(action, ImpulseAction)
        self.assertAlmostEqual(action.impulse_x, 6.0)
        self.assertAlmostEqual(action.impulse_y, 2.0)

    def test_impulse_left_and_down(self):
        """Impulse with Δvx < 0 and Δvy < 0."""
        action = parse_action(
            "Impulse la balle avec 4 Ns vers la gauche et 3 Ns vers le bas"
        )

        self.assertIsInstance(action, ImpulseAction)
        self.assertAlmostEqual(action.impulse_x, -4.0)
        self.assertAlmostEqual(action.impulse_y, -3.0)

    def test_different_values(self):
        """Parse two different values: 10 N along x, 5 N along y."""
        action = parse_action(
            "Pousse la balle avec 10 N vers la droite et 5 N vers le haut"
        )

        self.assertIsInstance(action, PushAction)
        self.assertAlmostEqual(action.force_x, 10.0)
        self.assertAlmostEqual(action.force_y, 5.0)

    def test_missing_component_defaults_to_zero(self):
        """An absent component stays at zero."""
        # Old horizontal syntax: no y component
        action = parse_action(
            "Pousse la balle vers la droite à 10 N"
        )

        self.assertIsInstance(action, PushAction)
        self.assertAlmostEqual(action.force_x, 10.0)
        self.assertAlmostEqual(action.force_y, 0.0)

    def test_old_vertical_syntax_still_valid(self):
        """Old vertical syntax still valid."""
        action = parse_action(
            "Pousse la balle à 15 N"
        )

        self.assertIsInstance(action, PushAction)
        self.assertAlmostEqual(action.force_x, 0.0)
        self.assertAlmostEqual(action.force_y, 15.0)

    def test_old_horizontal_syntax_still_valid(self):
        """Old horizontal syntax still valid."""
        action = parse_action(
            "Pousse la balle horizontalement à 10 N"
        )

        self.assertIsInstance(action, PushAction)
        self.assertAlmostEqual(action.force_x, 10.0)
        self.assertAlmostEqual(action.force_y, 0.0)

    def test_horizontal_impulse_preserves_support(self):
        """A positive vertical component can break the support."""
        world = build_supported_ball()

        # Impulse with a positive vertical component
        action = parse_action(
            "Impulse la balle avec 0 Ns vers la droite et 5 Ns vers le haut"
        )

        execute_action(world, action)

        # The support must be removed because impulse_y > 0
        self.assertFalse(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                for relation in world.relations
            )
        )


class DiagonalPhysicsTests(unittest.TestCase):
    """Tests for the physics of diagonal vectors."""

    def test_diagonal_force_simulation(self):
        """Diagonal force: vx and vy evolve correctly."""
        world = World()

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

        # Apply a diagonal force via language
        action = parse_action(
            "Pousse la balle avec 10 N vers la droite et 20 N vers le haut"
        )
        execute_action(world, action)

        simulate(
            world,
            laws=[ForceDynamicsLaw(providers=[ExternalForce()])],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # ax = 10/1 = 10 m/s² → vx = 10 m/s
        self.assertAlmostEqual(ball.properties["vx"], 10.0)

        # ay = 20/1 = 20 m/s² → vy = 20 m/s
        self.assertAlmostEqual(ball.properties["vy"], 20.0)


class WallCollisionTests(unittest.TestCase):
    """Tests for horizontal collisions with walls."""

    def test_no_wall_free_movement(self):
        """Without a wall: x evolves normally."""
        world = World()

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

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=10.0,
                    fy=0.0,
                    source="push",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # Without a wall, x must be 5.5 m (semi-implicit Euler)
        self.assertAlmostEqual(ball.properties["x"], 5.5)

    def test_right_wall_limits_x(self):
        """Right wall collision: x limited to max_x."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-10.0,
                    max_x=5.0,
                    restitution=0.0,
                )
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x must not exceed 5.0
        self.assertAlmostEqual(ball.properties["x"], 5.0)

        # vx must be 0 (restitution=0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_left_wall_limits_x(self):
        """Left wall collision: x limited to min_x."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": -4.8,
                    "y": 0.0,
                    "vx": -5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-5.0,
                    max_x=10.0,
                    restitution=0.0,
                )
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x must not be less than -5.0
        self.assertAlmostEqual(ball.properties["x"], -5.0)

        # vx must be 0 (restitution=0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_restitution_zero(self):
        """Restitution 0: vx becomes 0 after collision."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-10.0,
                    max_x=5.0,
                    restitution=0.0,
                )
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_restitution_one(self):
        """Restitution 1: perfectly elastic bounce."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 2.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-10.0,
                    max_x=5.0,
                    restitution=1.0,
                )
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x limited to 5.0
        self.assertAlmostEqual(ball.properties["x"], 5.0)

        # vx reversed with the same amplitude
        self.assertAlmostEqual(ball.properties["vx"], -2.0)

    def test_restitution_half(self):
        """Restitution 0.5: amplitude halved."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 4.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-10.0,
                    max_x=5.0,
                    restitution=0.5,
                )
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x limited to 5.0
        self.assertAlmostEqual(ball.properties["x"], 5.0)

        # vx reversed with reduced amplitude
        self.assertAlmostEqual(ball.properties["vx"], -2.0)

    def test_horizontal_collision_preserves_y_and_vy(self):
        """Horizontal collision: y and vy unchanged."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 10.0,
                    "vx": 5.0,
                    "vy": -9.81,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                GravityForce(),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-10.0,
                    max_x=5.0,
                    restitution=0.0,
                )
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x limited to 5.0
        self.assertAlmostEqual(ball.properties["x"], 5.0)

        # y must keep evolving normally under gravity
        # vy should be more negative
        self.assertNotAlmostEqual(ball.properties["y"], 10.0)

    def test_gravity_and_wall_collision(self):
        """Gravity + wall collision: vertical motion continues."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 10.0,
                    "vx": 5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                GravityForce(),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-10.0,
                    max_x=5.0,
                    restitution=0.0,
                )
            ],
            duration=0.5,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x limited to 5.0
        self.assertAlmostEqual(ball.properties["x"], 5.0)

        # y must have decreased because of gravity
        self.assertLess(ball.properties["y"], 10.0)

    def test_support_and_wall_coexistence(self):
        """Vertical support + wall: supported_by stays consistent."""
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
                    "x": -4.8,
                    "y": 0.0,
                    "vx": -5.0,
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

        law = ForceDynamicsLaw(
            providers=[
                GravityForce(),
                SupportForce(),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                GroundCollision(
                    ground_y=0.0,
                    restitution=0.0,
                    support_name="sol",
                ),
                WallCollision(
                    min_x=-5.0,
                    max_x=10.0,
                    restitution=0.0,
                ),
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x limited to -5.0
        self.assertAlmostEqual(ball.properties["x"], -5.0)

        # The ball must still be supported
        self.assertTrue(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                for relation in world.relations
            )
        )

    def test_no_wall_crossing_with_high_velocity(self):
        """Even at high velocity, the wall is not crossed."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 1000.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(
                    min_x=-10.0,
                    max_x=5.0,
                    restitution=0.0,
                )
            ],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        # x must never exceed 5.0
        self.assertAlmostEqual(ball.properties["x"], 5.0)


class WallEntityTests(unittest.TestCase):
    """Tests for walls as entities of the World."""

    def test_right_wall_as_entity(self):
        """Right wall created as an Entity."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                    "restitution": 0.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_left_wall_as_entity(self):
        """Left wall created as an Entity."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": -4.8,
                    "y": 0.0,
                    "vx": -5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_gauche",
                kind="wall",
                properties={
                    "x": -5.0,
                    "orientation": "vertical",
                    "side": "left",
                    "restitution": 0.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], -5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_wall_collision_detected_from_world(self):
        """Collision detected from the entities of the World."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_no_wall_entity_no_collision(self):
        """No wall in World: no artificial collision."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.5,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_wall_restitution_from_entity(self):
        """Restitution defined on the wall is respected."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                    "restitution": 0.5,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], -5.0)

    def test_add_wall_changes_behavior(self):
        """Adding a wall to World: the behaviour changes."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.8)

    def test_remove_wall_allows_crossing(self):
        """Removing a wall: the ball can pass through again."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        wall = Entity(
            name="mur_droit",
            kind="wall",
            properties={
                "x": 5.0,
                "orientation": "vertical",
            },
        )
        world.add_entity(wall)

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)

        world.entities.pop("mur_droit")

        ball.properties["vx"] = 10.0

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 6.0)

    def test_gravity_independent_of_wall_collision(self):
        """Gravity continues independently of the horizontal collision."""
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
                    "x": 4.8,
                    "y": 10.0,
                    "vx": 5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                    "restitution": 0.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                GravityForce(),
                SupportForce(),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                GroundCollision(ground_y=0.0, restitution=0.0),
                WallCollision(),
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)
        self.assertAlmostEqual(ball.properties["y"], 9.9019)

    def test_support_and_wall_entity_coexistence(self):
        """Vertical support stays consistent with a wall as an entity."""
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
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                },
            )
        )

        world.add_relation("balle", "supported_by", "sol")

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                GroundCollision(ground_y=0.0, restitution=0.0),
                WallCollision(),
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

        self.assertTrue(
            any(
                relation.subject == "balle"
                and relation.predicate == "supported_by"
                for relation in world.relations
            )
        )

    def test_both_walls_as_entities(self):
        """Both walls (left and right) as entities."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_gauche",
                kind="wall",
                properties={
                    "x": -5.0,
                    "orientation": "vertical",
                    "side": "left",
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=10.0,
                    fy=0.0,
                    source="push",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision()
            ],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_legacy_wall_collision_still_works(self):
        """The old API WallCollision(min_x, max_x) still works."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce(
                    "balle",
                    fx=0.0,
                    fy=0.0,
                    source="none",
                ),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[
                WallCollision(min_x=-5.0, max_x=5.0, restitution=0.0)
            ],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]

        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)


if __name__ == "__main__":
    unittest.main()
# This file contains the ColliderTests tests to add to test_wce.py
# Copy this content before the line "if __name__ == "__main__":


class ColliderTests(unittest.TestCase):
    """Tests for the VerticalPlaneCollider abstraction with Material."""

    def test_import_collider(self):
        """VerticalPlaneCollider can be imported."""
        from physics.colliders import Material, VerticalPlaneCollider
        collider = VerticalPlaneCollider(x=5.0, normal_x=-1.0)
        self.assertEqual(collider.x, 5.0)
        self.assertEqual(collider.normal_x, -1.0)

    def test_import_material(self):
        """Material can be imported."""
        from physics.colliders import DEFAULT_MATERIAL, Material
        mat = Material(restitution=0.5)
        self.assertEqual(mat.restitution, 0.5)
        self.assertEqual(DEFAULT_MATERIAL.restitution, 0.0)

    def test_collider_rejects_invalid_normal(self):
        """VerticalPlaneCollider rejects an invalid normal."""
        from physics.colliders import VerticalPlaneCollider
        with self.assertRaises(ValueError):
            VerticalPlaneCollider(x=5.0, normal_x=0.0)
        with self.assertRaises(ValueError):
            VerticalPlaneCollider(x=5.0, normal_x=2.0)

    def test_material_rejects_invalid_restitution(self):
        """Material rejects an invalid restitution."""
        from physics.colliders import Material
        with self.assertRaises(ValueError):
            Material(restitution=-0.1)
        with self.assertRaises(ValueError):
            Material(restitution=1.1)

    def test_right_wall_collider_no_side(self):
        """Right wall defined with VerticalPlaneCollider without side."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=0.0),
                    "fixed": True,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_left_wall_collider_no_side(self):
        """Left wall defined with VerticalPlaneCollider without side."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": -4.8,
                    "y": 0.0,
                    "vx": -5.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_gauche",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=-5.0, normal_x=+1.0),
                    "material": Material(restitution=0.0),
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], -5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_normal_direction_deduction(self):
        """The normal correctly determines the direction of the wall."""
        from physics.colliders import VerticalPlaneCollider

        right_collider = VerticalPlaneCollider(x=5.0, normal_x=-1.0)
        left_collider = VerticalPlaneCollider(x=-5.0, normal_x=+1.0)

        self.assertTrue(right_collider.is_right_facing)
        self.assertFalse(right_collider.is_left_facing)
        self.assertFalse(left_collider.is_right_facing)
        self.assertTrue(left_collider.is_left_facing)

    def test_right_wall_collision_with_collider(self):
        """Collision with a right wall using VerticalPlaneCollider."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=0.0),
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_left_wall_collision_with_collider(self):
        """Collision with a left wall using VerticalPlaneCollider."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": -4.8,
                    "y": 0.0,
                    "vx": -10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_gauche",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=-5.0, normal_x=+1.0),
                    "material": Material(restitution=0.0),
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], -5.0)

    def test_restitution_zero_with_collider_and_material(self):
        """Restitution 0 with VerticalPlaneCollider + Material."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=0.0),
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_restitution_one_with_collider_and_material(self):
        """Restitution 1 (elastic bounce) with VerticalPlaneCollider + Material."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=1.0),
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], -10.0)

    def test_restitution_half_with_collider_and_material(self):
        """Restitution 0.5 (damped bounce) with VerticalPlaneCollider + Material."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=0.5),
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], -5.0)

    def test_add_collider_changes_behavior(self):
        """Adding a collider to World modifies the behaviour."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.8)

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=0.0),
                },
            )
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_no_collider_no_artificial_collision(self):
        """No artificial collision without a collider."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.5,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_default_material_used_when_absent(self):
        """Default material used when material is absent."""
        from physics.colliders import VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    # No material -> DEFAULT_MATERIAL (restitution=0.0)
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_legacy_wall_format_still_works(self):
        """The old wall format (with side) still works via an adapter."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                    "restitution": 0.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_two_walls_confine_space(self):
        """Two walls (left and right) confine the space between -5 and +5."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 0.0,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_gauche",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=-5.0, normal_x=+1.0),
                    "material": Material(restitution=0.0),
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=0.0),
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=10.0, fy=0.0, source="push"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=1.0,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_modern_collider_with_contradictory_side_ignores_side(self):
        """Modern collider + contradictory side: collider wins."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=0.0),
                    "side": "left",
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)

    def test_modern_material_overrides_legacy_restitution(self):
        """Modern Material takes priority over legacy restitution."""
        from physics.colliders import Material, VerticalPlaneCollider

        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "collider": VerticalPlaneCollider(x=5.0, normal_x=-1.0),
                    "material": Material(restitution=1.0),
                    "restitution": 0.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], -10.0)

    def test_legacy_walls_with_side_still_work(self):
        """Legacy walls with side keep working."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                    "restitution": 0.0,
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_legacy_left_wall_converts_to_collider(self):
        """Legacy left wall (side=left) converted correctly."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": -4.8,
                    "y": 0.0,
                    "vx": -10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_gauche",
                kind="wall",
                properties={
                    "x": -5.0,
                    "orientation": "vertical",
                    "side": "left",
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], -5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_legacy_right_wall_converts_to_collider(self):
        """Legacy right wall (side=right) converted correctly."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                    "side": "right",
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)
        self.assertAlmostEqual(ball.properties["vx"], 0.0)

    def test_legacy_wall_without_side_deducted_from_x_position(self):
        """Legacy wall without side: direction deduced from x (x>=0→right, x<0→left)."""
        world = World()

        world.add_entity(
            Entity(
                name="balle",
                kind="objet",
                properties={
                    "x": 4.8,
                    "y": 0.0,
                    "vx": 10.0,
                    "vy": 0.0,
                    "masse": 1.0,
                },
            )
        )

        world.add_entity(
            Entity(
                name="mur_droit",
                kind="wall",
                properties={
                    "x": 5.0,
                    "orientation": "vertical",
                },
            )
        )

        law = ForceDynamicsLaw(
            providers=[
                ConstantForce("balle", fx=0.0, fy=0.0, source="none"),
            ]
        )

        simulate(
            world,
            laws=[law],
            interactions=[WallCollision()],
            duration=0.1,
            dt=0.1,
        )

        ball = world.entities["balle"]
        self.assertAlmostEqual(ball.properties["x"], 5.0)


class Transform2DTests(unittest.TestCase):
    """Tests for Transform2D."""

    def test_identity(self):
        """Identity transformation."""
        from core.transform import Transform2D
        t = Transform2D.identity()
        x, y = t.transform_point(5.0, 3.0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 3.0)

    def test_translation(self):
        """Translation: (1,2) -> (4,6)."""
        from core.transform import Transform2D
        t = Transform2D.translation(3.0, 4.0)
        x, y = t.transform_point(1.0, 2.0)
        self.assertAlmostEqual(x, 4.0)
        self.assertAlmostEqual(y, 6.0)

    def test_rotation_90_degrees(self):
        """Rotation 90°: (1,0) -> (0,1)."""
        from math import pi
        from core.transform import Transform2D
        t = Transform2D.rotation(pi / 2)
        x, y = t.transform_point(1.0, 0.0)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)

    def test_rotation_minus_90_degrees(self):
        """Rotation -90°: (0,1) -> (1,0)."""
        from math import pi
        from core.transform import Transform2D
        t = Transform2D.rotation(-pi / 2)
        x, y = t.transform_point(0.0, 1.0)
        self.assertAlmostEqual(x, 1.0, places=5)
        self.assertAlmostEqual(y, 0.0, places=5)

    def test_uniform_scale(self):
        """Uniform scale."""
        from core.transform import Transform2D
        t = Transform2D.scale(2.0)
        x, y = t.transform_point(3.0, 4.0)
        self.assertAlmostEqual(x, 6.0)
        self.assertAlmostEqual(y, 8.0)

    def test_non_uniform_scale(self):
        """Non-uniform scale."""
        from core.transform import Transform2D
        t = Transform2D.scale(2.0, 3.0)
        x, y = t.transform_point(1.0, 1.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 3.0)

    def test_inverse_translation(self):
        """Inverse of a translation."""
        from core.transform import Transform2D
        t = Transform2D.translation(5.0, 10.0)
        inv_t = t.inverse()
        x, y = inv_t.transform_point(7.0, 15.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 5.0)

    def test_inverse_identity(self):
        """T^-1(T(p)) = p."""
        from core.transform import Transform2D
        t = Transform2D.translation(5.0, 3.0)
        inv_t = t.inverse()
        
        for x, y in [(0, 0), (1, 2), (-3, 5)]:
            tx, ty = t.transform_point(x, y)
            ix, iy = inv_t.transform_point(tx, ty)
            self.assertAlmostEqual(ix, x, places=5)
            self.assertAlmostEqual(iy, y, places=5)

    def test_composition_order(self):
        """Composition order: T_final(p) = self(other(p))."""
        from core.transform import Transform2D
        t1 = Transform2D.translation(2.0, 0.0)
        t2 = Transform2D.translation(3.0, 0.0)
        
        # t2 after t1: total translation = 5
        t_final = t2.compose(t1)
        x, y = t_final.transform_point(0.0, 0.0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, 0.0)

    def test_vector_unchanged_by_translation(self):
        """A translation must not affect a vector."""
        from core.transform import Transform2D
        t = Transform2D.translation(10.0, 20.0)
        vx, vy = t.transform_vector(1.0, 2.0)
        self.assertAlmostEqual(vx, 1.0)
        self.assertAlmostEqual(vy, 2.0)

    def test_normal_transformed_correctly(self):
        """Normal transformed correctly (without non-uniform scale)."""
        from math import pi
        from core.transform import Transform2D
        t = Transform2D.rotation(pi / 2)
        nx, ny = t.transform_normal(1.0, 0.0)
        self.assertAlmostEqual(nx, 0.0, places=5)
        self.assertAlmostEqual(ny, 1.0, places=5)


class PlaneColliderTests(unittest.TestCase):
    """Tests for PlaneCollider."""

    def test_plane_collider_normalization(self):
        """PlaneCollider normalises the normal."""
        from physics.colliders import PlaneCollider
        pc = PlaneCollider(normal_x=2.0, normal_y=0.0, offset=0.0)
        self.assertAlmostEqual(pc.normal_x, 1.0)
        self.assertAlmostEqual(pc.normal_y, 0.0)

    def test_vertical_plane_collider_conversion_right(self):
        """VerticalPlaneCollider (right) converted to PlaneCollider."""
        from math import pi
        from core.transform import Transform2D
        from physics.colliders import VerticalPlaneCollider, vertical_plane_to_plane_collider
        
        vpc = VerticalPlaneCollider(x=5.0, normal_x=-1.0)
        pc = vertical_plane_to_plane_collider(vpc)
        
        # The PlaneCollider must have a translation of (5, 0)
        self.assertAlmostEqual(pc.transform.tx, 5.0)
        self.assertAlmostEqual(pc.transform.ty, 0.0)

    def test_vertical_plane_collider_conversion_left(self):
        """VerticalPlaneCollider (left) converted to PlaneCollider."""
        from physics.colliders import VerticalPlaneCollider, vertical_plane_to_plane_collider
        
        vpc = VerticalPlaneCollider(x=-5.0, normal_x=+1.0)
        pc = vertical_plane_to_plane_collider(vpc)
        
        self.assertAlmostEqual(pc.transform.tx, -5.0)
        self.assertAlmostEqual(pc.transform.ty, 0.0)

    def test_plane_collider_as_vertical_wall(self):
        """PlaneCollider used as a vertical wall."""
        from core.transform import Transform2D
        from physics.colliders import PlaneCollider
        
        # Right wall at x=5: local normal (1,0), transform=translation(5,0)
        pc = PlaneCollider(
            normal_x=1.0,
            normal_y=0.0,
            offset=0.0,
            transform=Transform2D.translation(5.0, 0.0),
        )
        
        # Point at x=6 is behind (normal side)
        self.assertTrue(pc.is_behind(6.0, 0.0))
        
        # Point at x=4 is not behind
        self.assertFalse(pc.is_behind(4.0, 0.0))
        
        # Point at x=5 is on the plane
        self.assertAlmostEqual(pc.signed_distance(5.0, 0.0), 0.0)

    def test_plane_collider_as_horizontal_floor(self):
        """PlaneCollider used as a horizontal floor."""
        from core.transform import Transform2D
        from physics.colliders import PlaneCollider
        
        # Floor at y=0: local normal (0,-1) to block y < 0 (below the floor)
        pc = PlaneCollider(
            normal_x=0.0,
            normal_y=-1.0,
            offset=0.0,
            transform=Transform2D.identity(),
        )
        
        # Point at y=-1 is behind (below the floor, forbidden side)
        self.assertTrue(pc.is_behind(0.0, -1.0))
        
        # Point at y=1 is not behind (above the floor)
        self.assertFalse(pc.is_behind(0.0, 1.0))
        
        # Point at y=0 is on the plane
        self.assertAlmostEqual(pc.signed_distance(0.0, 0.0), 0.0)

    def test_plane_collider_inclined(self):
        """PlaneCollider used as a plane inclined at 45°."""
        from math import pi
        from core.transform import Transform2D
        from physics.colliders import PlaneCollider
        
        # Plane inclined at 45°: local normal (0,1), rotation of 45°
        pc = PlaneCollider(
            normal_x=0.0,
            normal_y=1.0,
            offset=0.0,
            transform=Transform2D.rotation(pi / 4),
        )
        
        # 45° counter-clockwise rotation of (0,1):
        # nx = -sin(45°) ≈ -0.707, ny = cos(45°) ≈ 0.707
        nx, ny = pc.world_normal()
        self.assertAlmostEqual(nx, -0.70710678, places=5)
        self.assertAlmostEqual(ny, 0.70710678, places=5)

    def test_same_plane_collider_different_transforms(self):
        """Same PlaneCollider used as wall, floor and inclined plane with different Transform2D."""
        from math import pi
        from core.transform import Transform2D
        from physics.colliders import PlaneCollider
        
        # Same base PlaneCollider: local normal (1,0), offset=0
        base_plane = PlaneCollider(normal_x=1.0, normal_y=0.0, offset=0.0)
        
        # Right wall: translation to x=5
        wall = base_plane.__class__(
            normal_x=base_plane.normal_x,
            normal_y=base_plane.normal_y,
            offset=base_plane.offset,
            transform=Transform2D.translation(5.0, 0.0),
        )
        
        # Horizontal floor: rotation of 90° so that the normal points upwards
        floor = base_plane.__class__(
            normal_x=0.0,
            normal_y=1.0,
            offset=0.0,
            transform=Transform2D.identity(),
        )
        
        # Inclined plane: rotation of 45°
        inclined = base_plane.__class__(
            normal_x=0.0,
            normal_y=1.0,
            offset=0.0,
            transform=Transform2D.rotation(pi / 4),
        )
        
        # Check that each plane blocks in the right direction
        self.assertTrue(wall.is_behind(6.0, 0.0))  # Right wall blocks x > 5
        self.assertTrue(floor.is_behind(0.0, 1.0))  # Floor blocks y > 0
        self.assertTrue(inclined.is_behind(1.0, 1.0))  # Inclined plane blocks on the normal side



class TopologyTests(unittest.TestCase):
    """Tests for the topology module."""

    def test_topology_signature_validation(self):
        """Validation of the TopologySignature attributes."""
        from core.topology import TopologySignature
        
        # Valid
        ts = TopologySignature(connected_components=1, holes=0, boundary_components=1)
        self.assertEqual(ts.connected_components, 1)
        
        # Invalid: negative connected_components
        with self.assertRaises(ValueError):
            TopologySignature(connected_components=-1, holes=0, boundary_components=1)
        
        # Invalid: negative holes
        with self.assertRaises(ValueError):
            TopologySignature(connected_components=1, holes=-1, boundary_components=1)
    
    def test_topology_signature_is_solid(self):
        """Test of is_solid()."""
        from core.topology import TopologySignature
        
        solid = TopologySignature(connected_components=1, holes=0, boundary_components=1)
        self.assertTrue(solid.is_solid())
        
        ring = TopologySignature(connected_components=1, holes=1, boundary_components=2)
        self.assertFalse(ring.is_solid())
    
    def test_topology_signature_is_simple(self):
        """Test of is_simple()."""
        from core.topology import TopologySignature
        
        simple = TopologySignature(connected_components=1, holes=0, boundary_components=1)
        self.assertTrue(simple.is_simple())
        
        disconnected = TopologySignature(connected_components=2, holes=0, boundary_components=2)
        self.assertFalse(disconnected.is_simple())
        
        ring = TopologySignature(connected_components=1, holes=1, boundary_components=2)
        self.assertFalse(ring.is_simple())
    
    def test_combine_topologies(self):
        """Test of combine_topologies()."""
        from core.topology import TopologySignature, combine_topologies, DISK_TOPOLOGY
        
        # Two disjoint discs
        combined = combine_topologies(DISK_TOPOLOGY, DISK_TOPOLOGY)
        self.assertEqual(combined.connected_components, 2)
        self.assertEqual(combined.holes, 0)
        self.assertEqual(combined.boundary_components, 2)
    
    def test_preserved_topology_constants(self):
        """Test of the predefined topology constants."""
        from core.topology import (
            SEGMENT_TOPOLOGY, DISK_TOPOLOGY, CIRCLE_TOPOLOGY,
            RING_TOPOLOGY, DISJOINT_TWO_TOPOLOGY, POINT_TOPOLOGY,
        )
        
        self.assertTrue(SEGMENT_TOPOLOGY.is_simple())
        self.assertTrue(DISK_TOPOLOGY.is_simple())
        self.assertTrue(CIRCLE_TOPOLOGY.is_simple())
        self.assertFalse(RING_TOPOLOGY.is_simple())
        self.assertEqual(DISJOINT_TWO_TOPOLOGY.connected_components, 2)


class TopologyPreservationTests(unittest.TestCase):
    """Tests for TopologyPreservation."""

    def test_full_preservation(self):
        """Test of complete preservation."""
        from core.topology import TopologyPreservation
        
        full = TopologyPreservation.full()
        self.assertTrue(full.preserves_connected_components)
        self.assertTrue(full.preserves_holes)
        self.assertTrue(full.preserves_boundary)
        self.assertTrue(full.preserves_orientation)
    
    def test_continuous_preservation(self):
        """Test of continuous preservation."""
        from core.topology import TopologyPreservation
        
        continuous = TopologyPreservation.continuous()
        self.assertTrue(continuous.preserves_connected_components)
        self.assertTrue(continuous.preserves_holes)
        self.assertFalse(continuous.preserves_boundary)
        self.assertTrue(continuous.preserves_orientation)
    
    def test_reflection_preservation(self):
        """Test of preservation under reflection."""
        from core.topology import TopologyPreservation
        
        reflection = TopologyPreservation.reflection()
        self.assertTrue(reflection.preserves_connected_components)
        self.assertTrue(reflection.preserves_holes)
        self.assertTrue(reflection.preserves_boundary)
        self.assertFalse(reflection.preserves_orientation)


class DeformationTests(unittest.TestCase):
    """Tests for the deformation module."""

    def test_stretch_deformation(self):
        """Test of StretchDeformation."""
        from core.deformation import StretchDeformation
        
        stretch = StretchDeformation(stretch_x=2.0, stretch_y=3.0)
        x, y = stretch.apply_to_point(1.0, 1.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 3.0)
        
        # Check the transformation
        t = stretch.get_transform()
        x2, y2 = t.transform_point(1.0, 1.0)
        self.assertAlmostEqual(x2, 2.0)
        self.assertAlmostEqual(y2, 3.0)
    
    def test_identity_deformation(self):
        """Test of IdentityDeformation."""
        from core.deformation import IdentityDeformation
        
        identity = IdentityDeformation()
        x, y = identity.apply_to_point(5.0, -3.0)
        self.assertAlmostEqual(x, 5.0)
        self.assertAlmostEqual(y, -3.0)
    
    def test_compose_deformations(self):
        """Test of composition of deformations."""
        from core.deformation import StretchDeformation, compose_deformations
        
        stretch1 = StretchDeformation(stretch_x=2.0, stretch_y=1.0)
        stretch2 = StretchDeformation(stretch_x=1.0, stretch_y=3.0)
        
        composed = compose_deformations(stretch1, stretch2)
        x, y = composed.apply_to_point(1.0, 1.0)
        # stretch1 : (1, 1) -> (2, 1)
        # stretch2 : (2, 1) -> (2, 3)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 3.0)
    
    def test_stretch_topology_preservation(self):
        """Test of topological preservation for stretch."""
        from core.deformation import StretchDeformation
        
        # Positive stretch: preserves everything
        stretch = StretchDeformation(stretch_x=2.0, stretch_y=3.0)
        self.assertTrue(stretch.topology_preservation.preserves_orientation)
        
        # Stretch with a negative: reverses the orientation
        stretch_neg = StretchDeformation(stretch_x=-2.0, stretch_y=3.0)
        self.assertFalse(stretch_neg.topology_preservation.preserves_orientation)
    
    def test_circle_to_ellipse(self):
        """Test of the circle_to_ellipse function."""
        from core.deformation import circle_to_ellipse
        
        ellipse = circle_to_ellipse(2.0, 3.0)
        x, y = ellipse.apply_to_point(1.0, 1.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 3.0)


class GeometryKernelTests(unittest.TestCase):
    """Integration tests for the geometric kernel."""

    def test_transform2d_all_features(self):
        """Test of all the features of Transform2D."""
        from core.transform import Transform2D
        
        # Identity
        t_id = Transform2D.identity()
        self.assertEqual(t_id.tx, 0.0)
        self.assertEqual(t_id.ty, 0.0)
        self.assertEqual(t_id.rotation, 0.0)
        self.assertEqual(t_id.scale_x, 1.0)
        self.assertEqual(t_id.scale_y, 1.0)
        
        # Translation
        t_trans = Transform2D.translation(5.0, 10.0)
        x, y = t_trans.transform_point(1.0, 2.0)
        self.assertAlmostEqual(x, 6.0)
        self.assertAlmostEqual(y, 12.0)
        
        # Rotation 90°
        from math import pi
        t_rot = Transform2D.rotation(pi / 2)
        x, y = t_rot.transform_point(1.0, 0.0)
        self.assertAlmostEqual(x, 0.0, places=5)
        self.assertAlmostEqual(y, 1.0, places=5)
        
        # Scale
        t_scale = Transform2D.scale(2.0, 3.0)
        x, y = t_scale.transform_point(1.0, 1.0)
        self.assertAlmostEqual(x, 2.0)
        self.assertAlmostEqual(y, 3.0)
    
    def test_plane_collider_integration(self):
        """Integration test: PlaneCollider with WallCollision."""
        from core.transform import Transform2D
        from core.world import World, Entity
        from core.transformation import Transformation, Change
        from physics.colliders import PlaneCollider, Material
        from physics.interactions import WallCollision
        
        world = World()
        
        # Add a ball
        world.add_entity(Entity(
            name='balle',
            kind='ball',
            properties={'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'm': 1.0}
        ))
        
        # Add a right wall with PlaneCollider
        world.add_entity(Entity(
            name='mur_droit',
            kind='wall',
            properties={
                'collider': PlaneCollider(
                    normal_x=1.0,
                    normal_y=0.0,
                    offset=0.0,
                    transform=Transform2D.translation(5.0, 0.0)
                ),
                'material': Material(restitution=0.0)
            }
        ))
        
        # Transformation that moves the ball to the right
        transformation = Transformation(
            name='move',
            changes=[
                Change('balle', 'x', 0.0, 10.0),
                Change('balle', 'vx', 10.0, 10.0),
            ]
        )
        
        # Resolve the collisions
        wall_collision = WallCollision()
        resolved = wall_collision.resolve(world, transformation)
        
        # Check that the ball is stopped at x=5
        x_change = [c for c in resolved.changes if c.property == 'x'][0]
        self.assertAlmostEqual(x_change.after, 5.0)
        
        # Check that vx is set to 0
        vx_change = [c for c in resolved.changes if c.property == 'vx'][0]
        self.assertAlmostEqual(vx_change.after, 0.0)
    
    def test_plane_collider_as_floor(self):
        """Test of PlaneCollider used as a horizontal floor."""
        from core.transform import Transform2D
        from physics.colliders import PlaneCollider
        
        # Horizontal floor: local normal (0,1), no transform
        floor = PlaneCollider(
            normal_x=0.0,
            normal_y=1.0,
            offset=0.0,
            transform=Transform2D.identity(),
        )
        
        # Point above the floor (y=1) is behind
        self.assertTrue(floor.is_behind(0.0, 1.0))
        
        # Point below the floor (y=-1) is not behind
        self.assertFalse(floor.is_behind(0.0, -1.0))
        
        # Point on the floor (y=0) has distance 0
        self.assertAlmostEqual(floor.signed_distance(0.0, 0.0), 0.0)
    
    def test_plane_collider_inclined(self):
        """Test of PlaneCollider used as an inclined plane."""
        from math import pi
        from core.transform import Transform2D
        from physics.colliders import PlaneCollider
        
        # Plane inclined at 45°: local normal (0,1), rotation of 45°
        inclined = PlaneCollider(
            normal_x=0.0,
            normal_y=1.0,
            offset=0.0,
            transform=Transform2D.rotation(pi / 4),
        )
        
        # The world normal should be at 45°
        nx, ny = inclined.world_normal()
        # 45° counter-clockwise rotation of (0,1):
        # nx = cos(45°) * 0 - sin(45°) * 1 = -sin(45°) ≈ -0.707
        # ny = sin(45°) * 0 + cos(45°) * 1 = cos(45°) ≈ 0.707
        self.assertAlmostEqual(nx, -0.70710678, places=5)
        self.assertAlmostEqual(ny, 0.70710678, places=5)


class PlaneCollisionTests(unittest.TestCase):
    """Tests for PlaneCollision with PlaneCollider."""

    def test_plane_collision_vertical_wall(self):
        """Test of PlaneCollision with a vertical wall."""
        from core.transform import Transform2D
        from core.world import World, Entity
        from core.transformation import Transformation, Change
        from physics.colliders import PlaneCollider, Material
        from physics.geometry_collision import PlaneCollision
        
        world = World()
        
        # Add a ball
        world.add_entity(Entity(
            name='balle',
            kind='ball',
            properties={'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'm': 1.0}
        ))
        
        # Add a right wall with PlaneCollider
        world.add_entity(Entity(
            name='mur_droit',
            kind='wall',
            properties={
                'collider': PlaneCollider(
                    normal_x=1.0,  # Local normal pointing right
                    normal_y=0.0,
                    offset=0.0,
                    transform=Transform2D.translation(5.0, 0.0)
                ),
                'material': Material(restitution=0.0)
            }
        ))
        
        # Transformation that moves the ball to the right
        transformation = Transformation(
            name='move',
            changes=[
                Change('balle', 'x', 0.0, 10.0),
                Change('balle', 'vx', 10.0, 10.0),
            ]
        )
        
        # Resolve the collisions
        plane_collision = PlaneCollision()
        resolved = plane_collision.resolve(world, transformation)
        
        # Check that the ball is stopped at x=5 (on the plane)
        x_change = [c for c in resolved.changes if c.property == 'x'][0]
        self.assertAlmostEqual(x_change.after, 5.0, places=5)
        
        # Check that vx is set to 0 (restitution 0)
        vx_change = [c for c in resolved.changes if c.property == 'vx'][0]
        self.assertAlmostEqual(vx_change.after, 0.0, places=5)
    
    def test_plane_collision_horizontal_floor(self):
        """Test of PlaneCollision with a horizontal floor."""
        from core.transform import Transform2D
        from core.world import World, Entity
        from core.transformation import Transformation, Change
        from physics.colliders import PlaneCollider, Material
        from physics.geometry_collision import PlaneCollision
        
        world = World()
        
        # Add a ball above the floor
        world.add_entity(Entity(
            name='balle',
            kind='ball',
            properties={'x': 0.0, 'y': 5.0, 'vx': 0.0, 'vy': -10.0, 'm': 1.0}
        ))
        
        # Add a horizontal floor with PlaneCollider
        # Normal (0,-1) to block y < 0 (below the floor)
        world.add_entity(Entity(
            name='sol',
            kind='ground',
            properties={
                'collider': PlaneCollider(
                    normal_x=0.0,
                    normal_y=-1.0,  # Normal pointing down
                    offset=0.0,
                    transform=Transform2D.identity()
                ),
                'material': Material(restitution=0.5)
            }
        ))
        
        # Transformation that makes the ball fall
        transformation = Transformation(
            name='gravity',
            changes=[
                Change('balle', 'y', 5.0, -5.0),  # Fall to y=-5
                Change('balle', 'vy', -10.0, -10.0),
            ]
        )
        
        # Resolve the collisions
        plane_collision = PlaneCollision()
        resolved = plane_collision.resolve(world, transformation)
        
        # Check that the ball is stopped at y=0 (on the floor)
        y_change = [c for c in resolved.changes if c.property == 'y'][0]
        self.assertAlmostEqual(y_change.after, 0.0, places=5)
        
        # Check that vy bounces with restitution 0.5
        vy_change = [c for c in resolved.changes if c.property == 'vy'][0]
        # vy before = -10, after bounce: +10 * 0.5 = +5
        self.assertAlmostEqual(vy_change.after, 5.0, places=5)
    
    def test_plane_collision_inclined(self):
        """Test of PlaneCollision with a plane inclined at 45°."""
        from math import pi
        from core.transform import Transform2D
        from core.world import World, Entity
        from core.transformation import Transformation, Change
        from physics.colliders import PlaneCollider, Material
        from physics.geometry_collision import PlaneCollision
        
        world = World()
        
        # Add a ball
        world.add_entity(Entity(
            name='balle',
            kind='ball',
            properties={'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 10.0, 'm': 1.0}
        ))
        
        # Add a plane inclined at 45°
        # The local normal is (0,1) (local up)
        # With a rotation of 45°, the world normal is (-0.707, 0.707)
        # The plane passes through the origin
        # The forbidden side is y > x (because -x + y > 0)
        world.add_entity(Entity(
            name='plan_incline',
            kind='inclined_plane',
            properties={
                'collider': PlaneCollider(
                    normal_x=0.0,
                    normal_y=1.0,
                    offset=0.0,
                    transform=Transform2D.rotation(pi / 4)
                ),
                'material': Material(restitution=0.0)
            }
        ))
        
        # Transformation that moves the ball to (0, 10), which is on the forbidden side
        transformation = Transformation(
            name='move',
            changes=[
                Change('balle', 'x', 0.0, 0.0),
                Change('balle', 'y', 0.0, 10.0),
                Change('balle', 'vx', 0.0, 0.0),
                Change('balle', 'vy', 10.0, 10.0),
            ]
        )
        
        # Resolve the collisions
        plane_collision = PlaneCollision()
        resolved = plane_collision.resolve(world, transformation)
        
        # Check that the collision took place
        self.assertIn("+plane_collision", resolved.name)
        # Check that the ball is projected onto the plane y = x
        y_change = [c for c in resolved.changes if c.property == 'y'][0]
        x_change = [c for c in resolved.changes if c.property == 'x'][0]
        # On the plane y = x, so x = y
        self.assertAlmostEqual(x_change.after, y_change.after, places=5)
    
    def test_same_plane_collider_as_wall_floor_inclined(self):
        """Test: same PlaneCollider used as wall, floor and inclined plane."""
        from math import pi
        from core.transform import Transform2D
        from core.world import World, Entity
        from core.transformation import Transformation, Change
        from physics.colliders import PlaneCollider, Material
        from physics.geometry_collision import PlaneCollision
        
        # Same base PlaneCollider: local normal (0,1), offset=0
        base_plane = PlaneCollider(normal_x=0.0, normal_y=1.0, offset=0.0)
        
        # Scene 1: Used as a vertical wall at x=5
        # To block x > 5, the world normal must be (+1,0)
        # One can use local normal (+1,0) + translation(5,0)
        world1 = World()
        world1.add_entity(Entity(
            name='balle',
            kind='ball',
            properties={'x': 0.0, 'y': 0.0, 'vx': 10.0, 'vy': 0.0, 'm': 1.0}
        ))
        world1.add_entity(Entity(
            name='mur',
            kind='wall',
            properties={
                'collider': PlaneCollider(
                    normal_x=1.0,  # Local normal pointing right
                    normal_y=0.0,
                    offset=0.0,
                    transform=Transform2D.translation(5.0, 0.0)
                ),
                'material': Material(restitution=0.0)
            }
        ))
        
        trans1 = Transformation(
            name='move',
            changes=[
                Change('balle', 'x', 0.0, 10.0),
                Change('balle', 'vx', 10.0, 10.0),
            ]
        )
        
        plane_collision = PlaneCollision()
        resolved1 = plane_collision.resolve(world1, trans1)
        x_change1 = [c for c in resolved1.changes if c.property == 'x'][0]
        self.assertAlmostEqual(x_change1.after, 5.0, places=5)
        
        # Scene 2: Used as a horizontal floor at y=0
        # To block y < 0, the normal must be (0,-1)
        world2 = World()
        world2.add_entity(Entity(
            name='balle',
            kind='ball',
            properties={'x': 0.0, 'y': 5.0, 'vx': 0.0, 'vy': -10.0, 'm': 1.0}
        ))
        world2.add_entity(Entity(
            name='sol',
            kind='ground',
            properties={
                'collider': PlaneCollider(
                    normal_x=0.0,
                    normal_y=-1.0,  # Normal pointing down to block y < 0
                    offset=0.0,
                    transform=Transform2D.identity()
                ),
                'material': Material(restitution=0.0)
            }
        ))
        
        trans2 = Transformation(
            name='gravity',
            changes=[
                Change('balle', 'y', 5.0, -5.0),
                Change('balle', 'vy', -10.0, -10.0),
            ]
        )
        
        resolved2 = plane_collision.resolve(world2, trans2)
        y_change2 = [c for c in resolved2.changes if c.property == 'y'][0]
        self.assertAlmostEqual(y_change2.after, 0.0, places=5)
        
        # Scene 3: Used as a plane inclined at 45°
        # With local normal (0,1) and rotation 45°, the world normal is (-0.707, 0.707)
        # The plane passes through the origin
        # The forbidden side is the one where signed_distance > 0, that is:
        #   -0.707 * x + 0.707 * y > 0 => -x + y > 0 => y > x
        # So to test the collision, let us move the ball to (0, 10), which is in y > x
        world3 = World()
        world3.add_entity(Entity(
            name='balle',
            kind='ball',
            properties={'x': 0.0, 'y': 0.0, 'vx': 0.0, 'vy': 10.0, 'm': 1.0}
        ))
        world3.add_entity(Entity(
            name='plan_incline',
            kind='inclined_plane',
            properties={
                'collider': PlaneCollider(
                    normal_x=0.0,
                    normal_y=1.0,  # Local normal pointing up
                    offset=0.0,
                    transform=Transform2D.rotation(pi / 4)
                ),
                'material': Material(restitution=0.0)
            }
        ))
        
        trans3 = Transformation(
            name='move',
            changes=[
                Change('balle', 'x', 0.0, 0.0),
                Change('balle', 'y', 0.0, 10.0),
                Change('balle', 'vx', 0.0, 0.0),
                Change('balle', 'vy', 10.0, 10.0),
            ]
        )
        
        resolved3 = plane_collision.resolve(world3, trans3)
        # Check that the collision took place
        self.assertIn("+plane_collision", resolved3.name)
        # Check that the ball is projected onto the plane y = x
        y_change3 = [c for c in resolved3.changes if c.property == 'y'][0]
        x_change3 = [c for c in resolved3.changes if c.property == 'x'][0]
        # On the plane y = x, so x = y
        self.assertAlmostEqual(x_change3.after, y_change3.after, places=5)
        
        # Check that the same base PlaneCollider was used
        self.assertEqual(base_plane.normal_x, 0.0)
        self.assertEqual(base_plane.normal_y, 1.0)
        self.assertEqual(base_plane.offset, 0.0)
