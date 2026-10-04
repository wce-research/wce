from core.transformation import apply


def execute_action(world, action):
    transformation = action.compile(world)

    return apply(
        world,
        transformation,
    )
