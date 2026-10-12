"""Web-only resource boundaries. Research/core experiments retain their full history."""
from collections import deque
from dataclasses import fields, is_dataclass
import math
import threading
import time

from app.session import Session
from core.transform_graph import create_world_transform_graph
from language.command import AdvanceTimeCommand, parse_command
from physics.engine import step


class BudgetExceeded(ValueError):
    pass


def validate_numbers(value):
    if isinstance(value, float) and (not math.isfinite(value) or abs(value) > 1_000_000):
        raise ValueError('Numeric parameters must be finite and within ±1000000.')
    if is_dataclass(value):
        for field in fields(value):
            validate_numbers(getattr(value,field.name))


class LabSession(Session):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.world.history = deque(maxlen=config.history)
        self.commands = 0
        self.steps = 0
        self.checkpoints = 0
        self.created = time.monotonic()
        self.lock = threading.Lock()

    def checkpoint(self):
        graph = self.world.transform_graph
        if len(graph.nodes) < self.config.graph_nodes:
            return
        # Break old parent/child cycles; retain current world, not an invented full trace.
        for node in graph.nodes.values():
            node.parent = None
            node.children.clear()
        graph.nodes.clear()
        self.world.transform_graph = create_world_transform_graph('Web history checkpoint')
        self.world.transform_graph.root.configuration = self.world.get_state_snapshot()
        self.checkpoints += 1

    def execute(self, text):
        command = parse_command(text)
        validate_numbers(command)
        steps = 0
        if isinstance(command, AdvanceTimeCommand):
            if not math.isfinite(command.duration) or not 0 <= command.duration <= 60:
                raise ValueError('The maximum duration is 60 seconds.')
            steps = math.ceil(command.duration / self.dt)
        if self.commands >= self.config.commands or self.steps + steps > self.config.simulation_steps:
            raise BudgetExceeded('Laboratory session budget exhausted; start a new browser session or wait for expiry.')
        # Reserve work even if a scientific command is rejected later.
        self.commands += 1
        self.steps += steps
        self.checkpoint()
        if not isinstance(command, AdvanceTimeCommand):
            return super().execute(text)
        remaining = command.duration
        while remaining > 1e-12:
            self.checkpoint()
            current = min(self.dt, remaining)
            step(self.world,self.laws,current,self.interactions)
            remaining -= current
        return 'OK.'


class TokenBucket:
    """One process-wide budget: no growing per-IP registry or proxy trust requirement."""
    def __init__(self, rate, capacity):
        self.rate = rate
        self.capacity = capacity
        self.tokens = float(capacity)
        self.updated = time.monotonic()
        self.lock = threading.Lock()

    def take(self, cost=1):
        with self.lock:
            now = time.monotonic()
            self.tokens = min(self.capacity,self.tokens+(now-self.updated)*self.rate)
            self.updated = now
            if self.tokens < cost:
                return False
            self.tokens -= cost
            return True
