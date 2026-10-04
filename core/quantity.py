from dataclasses import dataclass


@dataclass(frozen=True)
class Quantity:
    value: float
    dimension: str
    unit: str


UNIT_TABLE = {
    # Longueur
    "m": ("length", 1.0, "m"),
    "cm": ("length", 0.01, "m"),
    "mm": ("length", 0.001, "m"),
    "km": ("length", 1000.0, "m"),

    # Force
    "n": ("force", 1.0, "N"),
    "kn": ("force", 1000.0, "N"),

    # Impulsion
    "ns": ("impulse", 1.0, "N·s"),
}


def make_quantity(value: float, unit: str) -> Quantity:
    key = unit.lower()

    if key not in UNIT_TABLE:
        raise ValueError(
            f"Unité inconnue : {unit!r}"
        )

    dimension, factor, canonical_unit = UNIT_TABLE[key]

    return Quantity(
        value=value * factor,
        dimension=dimension,
        unit=canonical_unit,
    )
