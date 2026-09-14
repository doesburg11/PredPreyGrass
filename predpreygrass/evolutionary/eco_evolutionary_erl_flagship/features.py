"""Hand-reduced, interpretable prey observation for Trial 13's evolved reward
genome -- an 8-channel feature vector echoing Trial 12's design
(`[visual_N, visual_S, visual_E, visual_W, in_tree, health_norm, energy_norm]`),
computed from flagship's own richer 4-channel spatial observation rather than a
newly hand-designed world model.

Two of Trial 12's 7 channels have no flagship equivalent and are deliberately
dropped: `in_tree` (flagship has no shelter mechanic) and a separate `health_norm`
(flagship prey have no health distinct from energy). See this module's README.md.

Deliberate coupling: `extract_prey_features` calls flagship's own
`env._get_observation` (predpreygrass_rllib_env.py) to get the exact same
egocentric, boundary-clipped crop the CNN policy already receives, rather than
re-deriving the windowing/clipping math independently -- so any behavioral gap
between genome-driven prey and PPO-trained prey is attributable to the
network/observation-compression difference, not withheld information or a
drifted-out-of-sync reimplementation. This does reach into a "private"
(underscore-prefixed) flagship method; if predpreygrass_rllib_env.py's
observation windowing changes shape or semantics, this module needs revisiting.
"""

import numpy as np

FEATURE_NAMES = [
    "energy_norm",
    "predator_dx",
    "predator_dy",
    "predator_proximity",
    "food_dx",
    "food_dy",
    "food_proximity",
    "local_grass_density",
]

# Channel order within env._get_observation's output, per config_env.py's own
# comment ("Border, Predator, Prey, Grass") and predpreygrass_rllib_env.py.
CH_BORDER, CH_PREDATOR, CH_PREY, CH_GRASS = 0, 1, 2, 3


def nearest_offset(channel: np.ndarray, half: int) -> tuple[float, float, float]:
    """(dx, dy, proximity) to the nearest nonzero cell in `channel`, relative to
    the window's center cell (index `half` on both axes -- true for flagship's
    odd-sized observation windows, see _obs_clip's observation_offset). dx/dy are
    signed offsets normalized by `half`; proximity is 1.0 at the center, 0.0 at
    the window edge. Returns (0.0, 0.0, 0.0) if no nonzero cell is visible.

    Public (not underscore-prefixed): also used by rule_based_predator.py to
    find the nearest visible prey, not just by extract_prey_features below."""
    rows, cols = np.nonzero(channel)
    if rows.size == 0:
        return 0.0, 0.0, 0.0
    d_rows = rows - half
    d_cols = cols - half
    cheby = np.maximum(np.abs(d_rows), np.abs(d_cols))
    nearest = int(np.argmin(cheby))
    dx = float(d_rows[nearest]) / half
    dy = float(d_cols[nearest]) / half
    proximity = 1.0 - float(cheby[nearest]) / half
    return dx, dy, proximity


def extract_prey_features(env, agent_id: str) -> np.ndarray:
    """8-dim feature vector (FEATURE_NAMES order) for one living prey agent."""
    obs = env._get_observation(agent_id)  # shape (4, obs_range, obs_range)
    half = obs.shape[1] // 2

    energy_norm = min(env.agent_energies[agent_id] / env.prey_creation_energy_threshold, 1.0)
    predator_dx, predator_dy, predator_proximity = nearest_offset(obs[CH_PREDATOR], half)
    food_dx, food_dy, food_proximity = nearest_offset(obs[CH_GRASS], half)

    in_bounds = obs[CH_BORDER] == 0
    local_grass_density = float((obs[CH_GRASS][in_bounds] > 0).mean()) if in_bounds.any() else 0.0

    return np.array(
        [
            energy_norm,
            predator_dx, predator_dy, predator_proximity,
            food_dx, food_dy, food_proximity,
            local_grass_density,
        ],
        dtype=np.float64,
    )
