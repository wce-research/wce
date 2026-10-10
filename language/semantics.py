from dataclasses import dataclass

from core.quantity import Quantity


@dataclass
class Intent:
    action: str
    entity: str
    quantity: Quantity | None = None
    direction: str | None = None
    # Vector components for 2D actions
    # If present, they take precedence over quantity/direction
    force_x: float | None = None
    force_y: float | None = None
    impulse_x: float | None = None
    impulse_y: float | None = None
