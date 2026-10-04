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

# Mots de conjonction pour les vecteurs 2D
CONJUNCTIONS = {
    "avec": "with",
    "et": "and",
}

# Mots de direction pour les axes
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
    Convertit une direction en axe et signe.
    Retourne (axis, sign) où axis est 'x' ou 'y' et sign est +1 ou -1.
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
        return ("y", +1)  # par défaut vertical positif


def _parse_component(tokens: list[str], start_index: int) -> tuple[float, str, int]:
    """
    Parse une composante de force/impulsion à partir des tokens.
    Retourne (value, direction, next_index).
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
        
        # Essayer de lire une valeur numérique
        if value is None:
            try:
                value = float(token)
                index += 1
                continue
            except ValueError:
                pass
        
        # Lire l'unité
        if unit is None and token in UNITS:
            unit = UNITS[token]
            index += 1
            continue
        
        # Lire la direction
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
        
        # Si on a une valeur mais pas d'unité, on suppose que c'est la fin de la composante
        if value is not None:
            break
        
        index += 1
    
    if value is None:
        raise ParseError(f"Valeur manquante pour la composante à {tokens[start_index]!r}")
    
    if unit is None:
        unit = "n"  # par défaut newton pour push/impulse
    
    # Normaliser la direction
    if direction is None:
        direction = "haut"  # par défaut vertical
    
    return (value, direction, index)


def extract_intent(text: str) -> Intent:
    tokens = tokenize(text)

    if not tokens:
        raise ParseError("Commande vide")

    canonical_tokens = [
        canonical(token)
        for token in tokens
    ]

    verb_token = canonical_tokens[0]

    if verb_token not in VERBS:
        raise ParseError(
            f"Verbe inconnu : {tokens[0]!r}"
        )

    action = VERBS[verb_token]

    entity = None
    numeric_value = None
    unit = None
    direction = None

    # Vérifier si c'est une phrase avec "avec ... et ..."
    # Pattern : "Pousse la balle avec 10 N vers la droite et 5 N vers le haut"
    if action in {"push", "impulse"}:
        # Trouver l'index de "avec" si présent
        try:
            with_index = canonical_tokens.index("avec")
        except ValueError:
            with_index = -1
        
        if with_index > 0:
            # Parser comme phrase vectorielle
            # Trouver l'index de "et"
            try:
                and_index = canonical_tokens.index("et", with_index + 1)
            except ValueError:
                and_index = -1
            
            if and_index > with_index:
                # Parser les deux composantes
                try:
                    value1, direction1, next_idx1 = _parse_component(
                        tokens, with_index + 1
                    )
                    value2, direction2, next_idx2 = _parse_component(
                        tokens, and_index + 1
                    )
                    
                    axis1, sign1 = _get_axis_sign(direction1)
                    axis2, sign2 = _get_axis_sign(direction2)
                    
                    # Créer l'Intent avec les composantes
                    intent = Intent(
                        action=action,
                        entity=None,  # sera rempli ci-dessous
                        quantity=None,
                        direction=None,
                    )
                    
                    # Déterminer l'entité (devrait être avant "avec")
                    # Parser l'entité entre le verbe et "avec"
                    for i in range(1, with_index):
                        token = canonical_tokens[i]
                        if token not in ARTICLES and token not in PREPOSITIONS:
                            entity = tokens[i]  # utiliser le token original
                            break
                    
                    if entity is None:
                        raise ParseError("Entité manquante")
                    
                    intent.entity = entity
                    
                    # Normaliser les unités (par défaut "n" pour force, "ns" pour impulsion)
                    default_unit = "n" if action == "push" else "ns"
                    
                    # Traiter la première composante
                    if axis1 == "x":
                        intent.force_x = value1 * sign1 if action == "push" else None
                        intent.impulse_x = value1 * sign1 if action == "impulse" else None
                    else:
                        intent.force_y = value1 * sign1 if action == "push" else None
                        intent.impulse_y = value1 * sign1 if action == "impulse" else None
                    
                    # Traiter la deuxième composante
                    if axis2 == "x":
                        intent.force_x = (intent.force_x or 0) + (value2 * sign2 if action == "push" else 0)
                        intent.impulse_x = (intent.impulse_x or 0) + (value2 * sign2 if action == "impulse" else 0)
                    else:
                        intent.force_y = (intent.force_y or 0) + (value2 * sign2 if action == "push" else 0)
                        intent.impulse_y = (intent.impulse_y or 0) + (value2 * sign2 if action == "impulse" else 0)
                    
                    return intent
                except (ValueError, IndexError):
                    # Si le parsing vectoriel échoue, continuer avec l'ancien parsing
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

        # Vérifier si c'est un mot de direction
        if token in DIRECTIONS:
            # Cas spécial : "vers la droite" ou "vers la gauche"
            if token == "vers":
                # Regarder le prochain token
                if index + 1 < len(canonical_tokens):
                    next_token = canonical_tokens[index + 1]
                    if next_token in ARTICLES:
                        # "vers la droite" ou "vers la gauche" ou "vers le haut" ou "vers le bas"
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
            f"Élément non compris : {tokens[index]!r}"
        )

    if entity is None:
        raise ParseError("Entité manquante")

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
                "Hauteur manquante pour l'action lift"
            )

        if intent.quantity.dimension != "length":
            raise ParseError(
                "L'action lift attend une longueur"
            )

        return LiftAction(
            entity_name=intent.entity,
            target_y=intent.quantity.value,
        )

    if intent.action == "push":
        # Si les composantes vectorielles sont déjà remplies, les utiliser
        if intent.force_x is not None or intent.force_y is not None:
            return PushAction(
                entity_name=intent.entity,
                force_y=intent.force_y or 0.0,
                force_x=intent.force_x or 0.0,
            )
        
        # Sinon, utiliser l'ancienne méthode (une seule quantité)
        if intent.quantity is None:
            raise ParseError(
                "Force manquante pour l'action push"
            )

        if intent.quantity.dimension != "force":
            raise ParseError(
                "L'action push attend une force"
            )

        magnitude = intent.quantity.value

        # Déterminer la direction
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
            # Par défaut : force verticale (comportement existant)
            return PushAction(
                entity_name=intent.entity,
                force_y=magnitude,
                force_x=0.0,
            )

    if intent.action == "impulse":
        # Si les composantes vectorielles sont déjà remplies, les utiliser
        if intent.impulse_x is not None or intent.impulse_y is not None:
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_y=intent.impulse_y or 0.0,
                impulse_x=intent.impulse_x or 0.0,
            )
        
        # Sinon, utiliser l'ancienne méthode (une seule quantité)
        if intent.quantity is None:
            raise ParseError(
                "Impulsion manquante"
            )

        if intent.quantity.dimension != "impulse":
            raise ParseError(
                "L'action impulse attend une impulsion"
            )

        magnitude = intent.quantity.value

        # Gérer la direction pour l'impulsion
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
            # Par défaut : impulsion verticale
            return ImpulseAction(
                entity_name=intent.entity,
                impulse_y=magnitude,
                impulse_x=0.0,
            )

    raise ParseError(
        f"Action inconnue : {intent.action!r}"
    )


def parse_action(text: str):
    intent = extract_intent(text)
    return compile_intent(intent)
