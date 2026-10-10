from core.action import (
    ImpulseAction,
    LiftAction,
    PushAction,
)
from core.quantity import make_quantity
from language.lexer import strip_accents, tokenize
from language.semantics import Intent

class ParseError(ValueError):
    pass


VERBS = {
    "souleve": "lift",
    "leve": "lift",
    "monte": "lift",
    "pousse": "push",
    "impulse": "impulse",
}

ARTICLES = {
    "le",
    "la",
    "les",
    "un",
    "une",
    "l",
}

PREPOSITIONS = {
    "a",
}

# Conjunction words for 2D vectors
CONJUNCTIONS = {
    "avec": "with",
    "et": "and",
}

# Direction words for the axes
DIRECTIONS = {
    "horizontalement": "x",
    "vers": "direction",
    "droite": "+x",
    "gauche": "-x",
    "haut": "+y",
    "bas": "-y",
}

UNITS = {
    "m": "m",
    "metre": "m",
    "metres": "m",
    "cm": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "mm": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "km": "km",
    "kilometre": "km",
    "kilometres": "km",
    "n": "n",
    "newton": "n",
    "newtons": "n",
    "kn": "kn",
    "ns": "ns",
}


def canonical(token: str) -> str:
    token = strip_accents(token)
    token = token.replace("'", "")
    return token


def _get_axis_sign(direction: str) -> tuple[str, float]:
    """
    Converts a direction into an axis and a sign.
    Returns (axis, sign) where axis is 'x' or 'y' and sign is +1 or -1.
    """
    if direction in {"+x", "droite", "x"}:
        return ("x", +1)
    elif direction in {"-x", "gauche"}:
        return ("x", -1)
    elif direction in {"+y", "haut", "y"}:
        return ("y", +1)
    elif direction in {"-y", "bas"}:
        return ("y", -1)
    else:
        return ("y", +1)  # positive vertical by default


def _parse_component(tokens: list[str], start_index: int) -> tuple[float, str, int]:
    """
    Parses a force/impulse component from the tokens.
    Returns (value, direction, next_index).
    """
    canonical_tokens = [canonical(token) for token in tokens]
    index = start_index
    
    value = None
    unit = None
    direction = None
    
    while index < len(canonical_tokens):
        token = canonical_tokens[index]
        
        if token in ARTICLES:
            index += 1
            continue
        
        if token in PREPOSITIONS:
            index += 1
            continue
        
        # Try to read a numeric value
        if value is None:
            try:
                value = float(token)
                index += 1
                continue
            except ValueError:
                pass
        
        # Read the unit
        if unit is None and token in UNITS:
            unit = UNITS[token]
            index += 1
            continue
        
        # Read the direction
        if token in DIRECTIONS:
            if token == "vers":
                if index + 1 < len(canonical_tokens):
                    next_token = canonical_tokens[index + 1]
                    if next_token in ARTICLES:
                        if index + 2 < len(canonical_tokens):
                            direction_token = canonical_tokens[index + 2]
                            if direction_token in {"droite", "gauche", "haut", "bas"}:
                                direction = direction_token
                                index += 3
                                continue
                    elif next_token in {"droite", "gauche", "haut", "bas"}:
                        direction = next_token
                        index += 2
                        continue
            elif token in {"droite", "gauche", "haut", "bas", "horizontalement"}:
                direction = token
                index += 1
                continue
        
        # If we have a value but no unit, we assume it is the end of the component
        if value is not None:
            break
        
        index += 1
    
    if value is None:
        raise ParseError(f"Missing value for the component at {tokens[start_index]!r}")
    
    if unit is None:
        unit = "n"  # newton by default for push/impulse
    
    # Normalise the direction
    if direction is None:
        direction = "haut"  # vertical by default
    
    return (value, direction, index)


def extract_intent(text: str) -> Intent:
    tokens = tokenize(text)

    if not tokens:
        raise ParseError("Empty command")

    canonical_tokens = [
        canonical(token)
        for token in tokens
    ]

    verb_token = canonical_tokens[0]

    if verb_token not in VERBS:
        raise ParseError(
            f"Unknown verb: {tokens[0]!r}"
        )

    action = VERBS[verb_token]

    entity = None
    numeric_value = None
    unit = None
    direction = None

    # Check whether it is a sentence with "avec ... et ..."
    # Pattern: "Pousse la balle avec 10 N vers la droite et 5 N vers le haut"
    if action in {"push", "impulse"}:
        # Find the index of "avec" if present
        try:
            with_index = canonical_tokens.index("avec")
        except ValueError:
            with_index = -1
        
        if with_index > 0:
            # Parse as a vector sentence
            # Find the index of "et"
            try:
                and_index = canonical_tokens.index("et", with_index + 1)
            except ValueError:
                and_index = -1
            
            if and_index > with_index:
                # Parse the two components
                try:
                    value1, direction1, next_idx1 = _parse_component(
                        tokens, with_index + 1
                    )
                    value2, direction2, next_idx2 = _parse_component(
                        tokens, and_index + 1
                    )
                    
                    axis1, sign1 = _get_axis_sign(direction1)
                    axis2, sign2 = _get_axis_sign(direction2)
                    
                    # Create the Intent with the components
                    intent = Intent(
                        action=action,
                        entity=None,  # will be filled in below
                        quantity=None,
                        direction=None,
                    )
                    
                    # Determine the entity (should be before "avec")
                    # Parse the entity between the verb and "avec"
                    for i in range(1, with_index):
                        token = canonical_tokens[i]
                        if token not in ARTICLES and token not in PREPOSITIONS:
                            entity = tokens[i]  # use the original token
                            break
                    
                    if entity is None:
                        raise ParseError("Missing entity")
                    
                    intent.entity = entity
                    
                    # Normalise the units ("n" by default for force, "ns" for impulse)
                    default_unit = "n" if action == "push" else "ns"
                    
                    # Process the first component
                    if axis1 == "x":
                        intent.force_x = value1 * sign1 if action == "push" else None
                        intent.impulse_x = value1 * sign1 if action == "impulse" else None
                    else:
                        intent.force_y = value1 * sign1 if action == "push" else None
                        intent.impulse_y = value1 * sign1 if action == "impulse" else None
                    
                    # Process the second component
                    if axis2 == "x":
                        intent.force_x = (intent.force_x or 0) + (value2 * sign2 if action == "push" else 0)
                        intent.impulse_x = (intent.impulse_x or 0) + (value2 * sign2 if action == "impulse" else 0)
                    else:
                        intent.force_y = (intent.force_y or 0) + (value2 * sign2 if action == "push" else 0)
                        intent.impulse_y = (intent.impulse_y or 0) + (value2 * sign2 if action == "impulse" else 0)
                    
                    return intent
                except (ValueError, IndexError):
                    # If the vector parsing fails, continue with the old parsing
                    pass

    index = 1

    while index < len(canonical_tokens):
        token = canonical_tokens[index]

        if token in ARTICLES:
            index += 1
            continue

        if token in PREPOSITIONS:
            index += 1
            continue

        # Check whether it is a direction word
        if token in DIRECTIONS:
            # Special case: "vers la droite" or "vers la gauche"
            if token == "vers":
                # Look at the next token
                if index + 1 < len(canonical_tokens):
                    next_token = canonical_tokens[index + 1]
                    if next_token in ARTICLES:
                        # "vers la droite" or "vers la gauche" or "vers le haut" or "vers le bas"
                        if index + 2 < len(canonical_tokens):
                            direction_token = canonical_tokens[index + 2]
                            if direction_token in {"droite", "gauche", "haut", "bas"}:
                                direction = direction_token
                                index += 3
                                continue
                    elif next_token in {"droite", "gauche", "haut", "bas"}:
                        direction = next_token
                        index += 2
                        continue
            elif token == "horizontalement":
                direction = "x"
                index += 1
                continue
            elif token in {"droite", "gauche", "haut", "bas"}:
                direction = token
                index += 1
                continue

        try:
            numeric_value = float(token)
            index += 1
            continue

        except ValueError:
            pass

        if token in UNITS:
            unit = UNITS[token]
            index += 1
            continue

        if entity is None:
            entity = token
            index += 1
            continue

        raise ParseError(
            f"Element not understood: {tokens[index]!r}"
        )

    if entity is None:
        raise ParseError("Missing entity")

    quantity = None

    if numeric_value is not None:
        if unit is None:
            unit = "m"

        try:
            quantity = make_quantity(
                numeric_value,
                unit,
            )
        except ValueError as error:
            raise ParseError(str(error)) from error

    return Intent(
        action=action,
        entity=entity,
        quantity=quantity,
        direction=direction,
    )


def compile_intent(intent: Intent):
    if intent.action == "lift":
        if intent.quantity is None:
            raise ParseError(
                "Missing height for the lift action"
            )

        if intent.quantity.dimension != "length":
            raise ParseError(
                "The lift action expects a length"
            )

        return LiftAction(
            entity_name=intent.entity,
            target_y=intent.quantity.value,
        )

    if intent.action == "push":
        # If the vector components are already filled in, use them
        if intent.force_x is not None or intent.force_y is not None:
            return PushAction(
                entity_name=intent.entity,
                force_y=intent.force_y or 0.0,
                force_x=intent.force_x or 0.0,
            )
        
        # Otherwise, use the old method (a single quantity)
        if intent.quantity is None:
            raise ParseError(
                "Missing force for the push action"
            )

        if intent.quantity.dimension != "force":
            raise ParseError(
                "The push action expects a force"
            )

        magnitude = intent.quantity.value

        # Determine the direction
        if intent.direction in {"x", "horizontalement", "+x", "droite"}:
            return PushAction(
                entity_name=intent.entity,
                force_x=magnitude,
                force_y=0.0,
            )
        elif intent.direction in {"-x", "gauche"}:
            return PushAction(
                entity_name=intent.entity,
                force_x=-magnitude,
                force_y=0.0,
            )
        elif intent.direction in {"+y", "haut"}:
            return PushAction(
                entity_name=intent.entity,
                force_x=0.0,
                force_y=magnitude,
            )
        elif intent.direction in {"-y", "bas"}:
            return PushAction(
                entity_name=intent.entity,
                force_x=0.0,
                force_y=-magnitude,
            )
        else:
            # By default: vertical force (existing behaviour)
            return PushAction(
                entity_name=intent.entity,
                force_y=magnitude,
                force_x=0.0,
            )

    if intent.action == "impulse":
        # If the vector components are already filled in, use them
        if intent.impulse_x is not None or intent.impulse_y is not None:
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_y=intent.impulse_y or 0.0,
                impulse_x=intent.impulse_x or 0.0,
            )
        
        # Otherwise, use the old method (a single quantity)
        if intent.quantity is None:
            raise ParseError(
                "Missing impulse"
            )

        if intent.quantity.dimension != "impulse":
            raise ParseError(
                "The impulse action expects an impulse"
            )

        magnitude = intent.quantity.value

        # Handle the direction for the impulse
        if intent.direction in {"+x", "droite", "x"}:
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_x=magnitude,
                impulse_y=0.0,
            )
        elif intent.direction in {"-x", "gauche"}:
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_x=-magnitude,
                impulse_y=0.0,
            )
        elif intent.direction in {"+y", "haut", "y"}:
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_x=0.0,
                impulse_y=magnitude,
            )
        elif intent.direction in {"-y", "bas"}:
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_x=0.0,
                impulse_y=-magnitude,
            )
        else:
            # By default: vertical impulse
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_y=magnitude,
                impulse_x=0.0,
            )

    raise ParseError(
        f"Unknown action: {intent.action!r}"
    )


def parse_action(text: str):
    intent = extract_intent(text)
    return compile_intent(intent)
