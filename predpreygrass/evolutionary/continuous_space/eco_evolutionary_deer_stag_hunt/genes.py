"""The hunters' innate attraction genes and the steering bias they add to
the move choice (DESIGN.md).

A predator's observation ends with its own genes (GENES). From the slots of
the hunters and deer it sees and its genes, the innate desire is

    d = a_hunter * mean(unit offsets of seen hunters)
      + a_young  * mean(unit offsets of seen deer below the adult energy)
      + a_adult  * mean(unit offsets of seen adult deer)

and move m (direction u_m, speed s_m) gets the bias beta * s_m * (u_m . d).
Only slots with the seen flag set count; a kind not in view adds nothing.
The deer's energy in the observation is divided by the deer's maximum
energy, so the adult threshold there is deer_adult_energy / energy_deer_max.
An observed energy is rounded to float32: a deer a hair below the threshold
can be read as adult (it never matters at the exact energies of the world).

The same bias is computed with NumPy (the E arm, evaluation) and with torch
(the PPO module, rl_module.py); tests check that the two agree.
"""

import math

import numpy as np

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.geometry import direction_vector

GENES = ("a_hunter", "a_young", "a_adult", "neutral")
DIRECTIONS = 16
SPEEDS = (1.0, 0.5, 0.0)  # as environment.SPEEDS
MOVES = DIRECTIONS * len(SPEEDS)
SLOT_SIZE = 6  # as environment.SLOT_SIZE: seen flag, dx, dy, distance, vx, vy


def move_table():
    """(MOVES, 2) unit directions and (MOVES,) speeds of the movement actions
    (environment._desired_velocity: direction move % 16, speed move // 16)."""
    directions = np.zeros((MOVES, 2))
    speeds = np.zeros(MOVES)
    for move in range(MOVES):
        v = direction_vector(move % DIRECTIONS, DIRECTIONS)
        directions[move] = (v.x, v.y)
        speeds[move] = SPEEDS[move // DIRECTIONS]
    return directions, speeds


DIRECTION_TABLE, SPEED_TABLE = move_table()
MOVING = SPEED_TABLE > 0


def layout(env):
    """Where the bias reads its inputs in a predator observation of env
    (an environment.EvolvingDeerStagHunt), as plain numbers (JSON-safe, for
    the module's model_config)."""
    return {
        "hunter_start": env.slot_start[("predator", "predator")],
        "hunter_count": env.observe[("predator", "predator")],
        "deer_start": env.slot_start[("predator", "deer")],
        "deer_count": env.observe[("predator", "deer")],
        "deer_energy_start": env.deer_energy_start,
        "gene_start": env.gene_start,
        "adult_threshold": float(np.float32(env.adult_energy / env.max_energy["deer"])),
        "obs_size": env.obs_size["predator"],
    }


def _mean_unit(slots, mask):
    """Mean unit offset over the masked slots; zero if none (numpy)."""
    offsets = slots[..., 1:3]
    norm = np.linalg.norm(offsets, axis=-1, keepdims=True)
    unit = np.divide(offsets, norm, out=np.zeros_like(offsets), where=norm > 0)
    weight = mask[..., None].astype(float)
    count = weight.sum(axis=-2)
    return np.divide((unit * weight).sum(axis=-2), count, out=np.zeros_like(count * unit[..., 0, :]), where=count > 0)


def desire(obs, lay):
    """Innate desire d, (n, 2), from (n, obs_size) predator observations."""
    obs = np.asarray(obs, dtype=float).reshape(-1, lay["obs_size"])
    hunters = obs[:, lay["hunter_start"] : lay["hunter_start"] + SLOT_SIZE * lay["hunter_count"]].reshape(len(obs), lay["hunter_count"], SLOT_SIZE)
    deer = obs[:, lay["deer_start"] : lay["deer_start"] + SLOT_SIZE * lay["deer_count"]].reshape(len(obs), lay["deer_count"], SLOT_SIZE)
    energy = obs[:, lay["deer_energy_start"] : lay["deer_energy_start"] + lay["deer_count"]]
    genes = obs[:, lay["gene_start"] : lay["gene_start"] + len(GENES)]
    seen_hunter = hunters[..., 0] > 0.5
    seen_deer = deer[..., 0] > 0.5
    adult = seen_deer & (energy >= lay["adult_threshold"])
    young = seen_deer & ~adult
    return genes[:, 0:1] * _mean_unit(hunters, seen_hunter) + genes[:, 1:2] * _mean_unit(deer, young) + genes[:, 2:3] * _mean_unit(deer, adult)


def bias(obs, lay, beta):
    """(n, MOVES) innate bias of each move (numpy)."""
    return beta * SPEED_TABLE[None, :] * (desire(obs, lay) @ DIRECTION_TABLE.T)


def anything_in_view(obs, lay):
    """(n,) whether the predator sees any hunter or deer."""
    obs = np.asarray(obs, dtype=float).reshape(-1, lay["obs_size"])
    flags = [obs[:, lay["hunter_start"] + SLOT_SIZE * k] for k in range(lay["hunter_count"])]
    flags += [obs[:, lay["deer_start"] + SLOT_SIZE * k] for k in range(lay["deer_count"])]
    return np.any(np.stack(flags, axis=1) > 0.5, axis=1) if flags else np.zeros(len(obs), dtype=bool)


def cruise_move(obs):
    """(n,) the full-speed move closest to each predator's current heading
    (observation entries 2, 3: sin and cos of the heading, counterclockwise
    from +x with y pointing down, so the heading's direction is
    (cos h, -sin h))."""
    obs = np.asarray(obs, dtype=float)
    heading = np.stack([obs[:, 3], -obs[:, 2]], axis=1)
    scores = heading @ DIRECTION_TABLE[:DIRECTIONS].T
    return np.argmax(scores, axis=1)


def e_arm_logits(obs, lay, beta):
    """The E arm's move logits (DESIGN.md, three arms): with nothing in view,
    certainty of the cruise move; otherwise the bias with stop moves masked
    out. Returns (n, MOVES) with -inf for excluded moves."""
    obs = np.asarray(obs, dtype=float).reshape(-1, lay["obs_size"])
    logits = bias(obs, lay, beta)
    logits[:, ~MOVING] = -np.inf
    blind = ~anything_in_view(obs, lay)
    if blind.any():
        cruise = cruise_move(obs[blind])
        rows = np.full((blind.sum(), MOVES), -np.inf)
        rows[np.arange(len(cruise)), cruise] = 0.0
        logits[blind] = rows
    return logits


def sample(logits, rng):
    """One move per row from softmax(logits) (numpy Generator rng)."""
    logits = np.asarray(logits, dtype=float)
    top = logits.max(axis=1, keepdims=True)
    p = np.exp(logits - top)
    p /= p.sum(axis=1, keepdims=True)
    u = rng.random(len(p))[:, None]
    return np.minimum((p.cumsum(axis=1) < u).sum(axis=1), MOVES - 1)


def beta_rule_probability(beta):
    """DESIGN.md's rule for beta: a hunter with one gene at +1 and one
    target of that kind in view straight along move direction 0, network
    logits zero: the probability of a full-speed move within 22.5 degrees of
    the target's direction."""
    d = DIRECTION_TABLE[0]
    logits = beta * SPEED_TABLE * (DIRECTION_TABLE @ d)
    p = np.exp(logits - logits.max())
    p /= p.sum()
    near = (SPEED_TABLE == 1.0) & (DIRECTION_TABLE @ d >= math.cos(math.radians(22.5)) - 1e-9)
    return float(p[near].sum())


def solve_beta(target=0.5):
    """The beta at which beta_rule_probability is target (bisection)."""
    low, high = 0.0, 50.0
    for _ in range(200):
        mid = (low + high) / 2
        if beta_rule_probability(mid) < target:
            low = mid
        else:
            high = mid
    return (low + high) / 2


BETA0 = solve_beta()
