# SELFish: escape-based flocking

This is a separate implementation of Hahn, Phan, Gabor, Belzner, and
Linnhoff-Popien (2019), *Emergent Escape-based Flocking Behavior using
Multi-Agent Reinforcement Learning*. It does not use the
Predator-Prey-Grass environment: there is one predator, a fixed population of
prey, no grass, no births, and no energy.

## Paper setup implemented

- Toroidal continuous arena; training defaults to 40 × 40 with 10 prey.
- Prey have radius 1, move at constant speed 1, and do not collide with each
  other. The predator catches a prey at distance less than 2. Caught prey
  respawn at a random position to preserve population size.
- The predator pursues prey, turns at most 45 degrees per frame, selects a
  random target when multiple prey are nearby, and occasionally makes a
  faster leap. Its target-switch threshold, target hold, leap schedule, and
  leap speed are **not numerically specified in the paper**; the values in
  `Settings` are explicit implementation assumptions, not verified paper
  parameters.
- A prey observation is a fixed sequence of `(distance, relative bearing,
  absolute heading)` rows for the predator, self, and nearest conspecifics.
  Distances and angles are normalized to [0, 1]. DQN observes five
  conspecifics; DDPG observes one, matching the paper's hyperparameter table.
- DQN has five turn actions (`-90, -45, 0, 45, 90` degrees); DDPG outputs a
  continuous turn bounded by ±45 degrees. Prey receive +1 per living frame
  and -1000 when caught. Each episode ends on the focal prey's death or after
  10,000 frames. Other prey follow a frozen copy of the shared policy, which
  is updated for the cohort after the episode.
- `train.py` uses 500,000 training frames by default and the algorithm
  hyperparameters reported in the paper: DQN (10 × 16 network, gamma
  0.999999, Adam 0.001, replay 50,000, batch 64, epsilon 0.1, five neighbors)
  or DDPG (five hidden layers, 16 actor / 32 critic units, gamma 0.999999,
  Adam 0.001, replay 100,000, batch 512, OU noise, one neighbor).

The paper does not specify every implementation detail (including the
predator's numeric behavior settings, exact angle/distance normalization,
target-network schedule, or the final Boids weights). Those are visible and
saved alongside each run rather than silently presented as exact matches.
This is therefore a reproducible, paper-informed implementation—not a
bit-for-bit reproduction of the authors' Keras-RL code.

## Run

From the repository root:

```bash
python -m predpreygrass.non_evolutionary.continuous_space.selfish_flocking.train \
    --algo dqn --steps 500000 --seed 1
```

Use `--algo ddpg` for the continuous-action learner. By default checkpoints
are written under `selfish_flocking/runs/`; `--out-dir` changes the location.
Evaluation runs the paper's Random, TurnAway, and Boids comparison policies.
To include a trained policy, pass its checkpoint:

```bash
python -m predpreygrass.non_evolutionary.continuous_space.selfish_flocking.evaluate \
    --checkpoint selfish_flocking/runs/dqn_step_00500000.pt \
    --algo dqn --prey-count 40 --episodes 20 --seed 100
```

The evaluator reports focal survival, catch rate, pairwise spacing, cluster
count, noise agents, and within-cluster heading deviation. The paper uses
DBSCAN for some figures but does not give all clustering parameters; this
implementation reports connected components at a configurable `--cluster-radius`
and labels that threshold in the output. The Boids weights are also explicit
approximations because their tuned values are not reported numerically.

## Validation limits

The paper trained in 40 × 40 with 10 prey and evaluated shared policies in
larger populations. Evaluation population sizes and repeated runs should be
chosen explicitly; one seed or one learned checkpoint is not enough to claim
the paper's aggregate result. The provided tests cover deterministic reset,
torus geometry, action handling, bounded movement, and the DQN update path.

Reference: [arXiv:1905.04077](https://arxiv.org/abs/1905.04077).
