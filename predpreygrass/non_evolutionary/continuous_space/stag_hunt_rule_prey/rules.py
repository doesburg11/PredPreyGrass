"""The prey's fixed rules: no learning, the same behaviour in every run.

At each decision a prey flees the nearest predator in its view cone within its
flee distance, at full speed. Otherwise it goes to the nearest edible patch of
its food in view (full speed, stopping when it can eat), or else wanders at
half speed, turning at most one direction step at a time.
"""

import math

DIRECTIONS = 16
SPEEDS = (1.0, 0.5, 0.0)  # full, half, stopped
FULL, HALF, STOPPED = range(len(SPEEDS))
PREY_FOOD = {"rabbit": "grass", "mammoth": "steppe"}


def direction_towards(dx, dy):
    """The movement direction (geometry.direction_vector: 0 points to -y,
    increasing clockwise) nearest to the vector (dx, dy)."""
    return round(math.degrees(math.atan2(dx, -dy)) / (360 / DIRECTIONS)) % DIRECTIONS


def move_of(direction, speed):
    return direction + DIRECTIONS * speed


def prey_move(env, prey, rng):
    """A prey's movement action in this state. Draws from rng only to wander."""
    threats = []
    for predator in env.predators:
        if env.torus.in_view(prey, predator.position, env.flee_distance[prey.species], prey.fov):
            dx, dy = env.torus.offset(prey.position, predator.position)
            threats.append((math.hypot(dx, dy), predator.id, dx, dy))
    if threats:
        _, _, dx, dy = min(threats)
        return move_of(direction_towards(-dx, -dy), FULL)
    food = env.food[PREY_FOOD[prey.species]]
    seen, dx, dy, distance = food.observation(prey, prey.view_distance, prey.fov)
    if seen:
        if distance * prey.view_distance <= food.consume_radius:
            return move_of(0, STOPPED)
        return move_of(direction_towards(float(dx), float(dy)), FULL)
    if prey.velocity.mag() > 0:
        heading = direction_towards(prey.velocity.x, prey.velocity.y)
    else:
        heading = rng.randrange(DIRECTIONS)
    return move_of((heading + rng.choice((-1, 0, 1))) % DIRECTIONS, HALF)
