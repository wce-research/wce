from language.query import Query


class AnswerError(ValueError):
    pass


def answer_query(world, query: Query) -> str:
    if query.kind == "time":
        return (
            f"Le temps du monde est de "
            f"{format_number(world.time)} s."
        )

    if query.entity not in world.entities:
        raise AnswerError(
            f"Entité inconnue : {query.entity!r}"
        )

    entity = world.entities[query.entity]

    if query.kind == "height":
        if "y" not in entity.properties:
            raise AnswerError(
                f"Aucune hauteur connue pour {query.entity!r}"
            )

        y = entity.properties["y"]

        return (
            f"La hauteur de {query.entity} "
            f"est de {format_number(y)} m."
        )

    if query.kind == "x_position":
        if "x" not in entity.properties:
            raise AnswerError(
                f"Aucune position horizontale connue pour {query.entity!r}"
            )

        x = entity.properties["x"]

        return (
            f"La position horizontale de {query.entity} "
            f"est de {format_number(x)} m."
        )

    if query.kind == "vx":
        if "vx" not in entity.properties:
            raise AnswerError(
                f"Aucune vitesse horizontale connue pour {query.entity!r}"
            )

        vx = entity.properties["vx"]

        return (
            f"La vitesse horizontale de {query.entity} "
            f"est de {format_number(vx)} m/s."
        )

    if query.kind == "vy":
        if "vy" not in entity.properties:
            raise AnswerError(
                f"Aucune vitesse verticale connue pour {query.entity!r}"
            )

        vy = entity.properties["vy"]

        return (
            f"La vitesse verticale de {query.entity} "
            f"est de {format_number(vy)} m/s."
        )

    if query.kind == "velocity":
        vx = entity.properties.get("vx")
        vy = entity.properties.get("vy")

        if vx is None and vy is None:
            raise AnswerError(
                f"Aucune vitesse connue pour {query.entity!r}"
            )

        parts = []

        if vx is not None:
            parts.append(f"vx={format_number(vx)} m/s")

        if vy is not None:
            parts.append(f"vy={format_number(vy)} m/s")

        if parts:
            return (
                f"{capitalize(query.entity)} "
                f"a une vitesse ({', '.join(parts)})."
            )

        raise AnswerError(
            f"Vitesse inconnue pour {query.entity!r}"
        )

    if query.kind == "position":
        x = entity.properties.get("x")
        y = entity.properties.get("y")

        if x is None and y is None:
            raise AnswerError(
                f"Aucune position connue pour {query.entity!r}"
            )

        # Vérifier le support d'abord
        support = find_relation_object(
            world,
            subject=query.entity,
            predicate="supported_by",
        )

        if support is not None:
            # Sur un support, y est probablement 0 ou connu
            if x is not None and x != 0:
                return (
                    f"{capitalize(query.entity)} "
                    f"est sur {support} à x={format_number(x)} m."
                )
            return (
                f"{capitalize(query.entity)} "
                f"est sur {support}."
            )

        # Construire la réponse avec x et y
        parts = []

        if x is not None:
            parts.append(f"x={format_number(x)} m")

        if y is not None:
            parts.append(f"y={format_number(y)} m")

        if parts:
            return (
                f"{capitalize(query.entity)} "
                f"se trouve en ({', '.join(parts)})."
            )

        raise AnswerError(
            f"Position inconnue pour {query.entity!r}"
        )

    if query.kind == "location":
        support = find_relation_object(
            world,
            subject=query.entity,
            predicate="supported_by",
        )

        if support is not None:
            return (
                f"{capitalize(query.entity)} "
                f"est sur {support}."
            )

        if "y" in entity.properties:
            y = entity.properties["y"]

            return (
                f"{capitalize(query.entity)} "
                f"se trouve à {format_number(y)} m "
                f"de hauteur."
            )

        raise AnswerError(
            f"Position inconnue pour {query.entity!r}"
        )

    raise AnswerError(
        f"Type de question inconnu : {query.kind!r}"
    )


def find_relation_object(
    world,
    subject: str,
    predicate: str,
):
    for relation in world.relations:
        if (
            relation.subject == subject
            and relation.predicate == predicate
        ):
            return relation.object

    return None


def format_number(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))

    return (
        f"{value:.6f}"
        .rstrip("0")
        .rstrip(".")
    )


def capitalize(text: str) -> str:
    if not text:
        return text

    return text[0].upper() + text[1:]
