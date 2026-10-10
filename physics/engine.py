from core.transformation import apply


def step(
    world,
    laws,
    dt: float,
    interactions=None,
):
    interactions = interactions or []

    for law in laws:
        transformation = law.propose(world, dt)

        if transformation is None:
            continue

        for interaction in interactions:
            transformation = interaction.resolve(
                world,
                transformation,
            )

        apply(world, transformation)

    world.time += dt


def simulate(
    world,
    laws,
    duration: float,
    dt: float,
    interactions=None,
):
    if duration < 0:
        raise ValueError(
            "The duration cannot be negative"
        )

    if dt <= 0:
        raise ValueError(
            "dt must be strictly positive"
        )

    remaining = duration
    epsilon = 1e-12

    while remaining > epsilon:
        current_dt = min(
            dt,
            remaining,
        )

        step(
            world,
            laws,
            current_dt,
            interactions=interactions,
        )

        remaining -= current_dt
