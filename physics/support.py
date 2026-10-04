from core.transformation import Change, Transformation


def is_supported(world, entity_name: str) -> bool:
    for relation in world.relations:
        if (
            relation.subject == entity_name
            and relation.predicate == "supported_by"
        ):
            return True

    return False


def add_support(
    world,
    entity_name: str,
    support_name: str,
):
    for relation in world.relations:
        if (
            relation.subject == entity_name
            and relation.predicate == "supported_by"
            and relation.object == support_name
        ):
            return

    world.add_relation(
        entity_name,
        "supported_by",
        support_name,
    )


def remove_support(
    world,
    entity_name: str,
):
    world.relations = [
        relation
        for relation in world.relations
        if not (
            relation.subject == entity_name
            and relation.predicate == "supported_by"
        )
    ]
