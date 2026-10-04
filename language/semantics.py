from dataclasses import dataclass

from core.quantity import Quantity


@dataclass
class Intent:
    action: str
    entity: str
    quantity: Quantity | None = None
    direction: str | None = None
    # Composantes vectorielles pour les actions 2D
    # Si présentes, priment sur quantity/direction
    force_x: float | None = None
    force_y: float | None = None
    impulse_x: float | None = None
    impulse_y: float | None = None
