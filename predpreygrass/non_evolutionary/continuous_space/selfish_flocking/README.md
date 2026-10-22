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
the paper's aggregate result. `nash_mixed.py` runs the mixed-population test; `--split k --records-dir DIR`
runs one split per process and a later call with only `--records-dir` merges
them (identical to a sequential run). The provided tests cover deterministic reset,
torus geometry, action handling, bounded movement, and the DQN update path.

## Results (2026-10-22)

All runs use the paper's training scale: 500,000 frames, 40 × 40, 10 prey,
paper hyperparameters. Five DQN seeds and three DDPG seeds were trained in
parallel (about 15 and 30 minutes each on one CPU core). Evaluation used 10
prey on 40 × 40, 20 episodes of 10,000 frames, evaluation seed 100. Raw
output is in `runs/eval/` (git-ignored).

### Training

| Run | Survival per training episode, 100k → 500k frames |
| --- | --- |
| DQN seed 4 | 98 → 593 (clear learning) |
| DQN seed 3 | 141 → 225 (noisy) |
| DQN seeds 1, 2, 5 | 105–135 throughout (no learning) |
| DDPG seeds 1–3 | 300–690, non-monotonic |

A random policy survives about 130 frames per episode, so only one DQN seed
in five learned clearly. Training survival includes ε = 0.1 exploration.

### Homogeneous populations

| Policy | Focal survival (frames) | Catches / 1000 frames | Groups (≥ 2) | Prey alone | Heading spread in group (°) |
| --- | --- | --- | --- | --- | --- |
| Random | 114–157 | 82 | 1.2 | 7.5 | 47 |
| TurnAway | 5,499 | 3.8 | 1.7 | 2.0 | 28 |
| Boids (discrete turns) | 1,984 | 6.6 | 1.3 | 1.4 | 28 |
| Boids (continuous turns) | 3,560 | 5.3 | 3.3 | 0.6 | 19 |
| SELFish DQN, seed 4 | 2,552 | 5.7 | 2.2 | 3.5 | 29 |
| SELFish DQN, seeds 1, 2, 3, 5 | 80–233 | 60–113 | 1.2–1.5 | 6.5–7.4 | 45–48 |
| SELFish DDPG, seeds 1–3 | 1,188–1,402 | 10.8–11.6 | 1.5–1.9 | 2.7–6.1 | 29 |

Groups are connected components at radius 4. TurnAway surviving longest
matches the paper. Learned policies that work (DQN seed 4, all DDPG seeds)
group more, and with more aligned headings, than random movement. TurnAway
groups just as much while ignoring other prey entirely: everyone fleeing the
same predator ends up running together. Grouping alone is therefore not
evidence of learned social behavior in this task.

The first Boids implementation was broken (it averaged absolute neighbour
headings in the prey's own frame and let an inverse-square separation term
on normalized distances dominate) and did worse than random. The rows above
are from the corrected version; its weights and radii are assumptions.

### Mixed populations (Nash test, ICAART 2020)

`nash_mixed.py` plays every split of k TurnAway and 10 − k SELFish prey and
checks whether any single prey gains by switching strategy. Full paper
exposure for one checkpoint (DQN seed 4: 10 episodes × 100,000 frames per
split, 11 splits run in parallel and merged):

| TurnAway k | TurnAway frames per catch | SELFish frames per catch [95% CI] | Share of bootstrap draws stable |
| --- | --- | --- | --- |
| 0 | – | 1,734 [1,709, 1,757] | 0.00 |
| 1 | 200,000 | 1,543 [1,530, 1,556] | 0.00 |
| 3 | 500,000 | 1,163 [1,148, 1,176] | 0.00 |
| 5 | 263,158 | 791 [782, 799] | 0.00 |
| 7 | 142,857 | 446 [442, 450] | 0.00 |
| 9 | 24,590 | 155 [153, 157] | 0.00 |
| 10 | 2,631 | – | 1.00 |

The only pure equilibrium is all-TurnAway (10, 0). A TurnAway prey in a mixed
population is almost never caught, while every SELFish prey's survival falls
as SELFish prey become scarcer, so switching to TurnAway always pays. The
reduced test (20,000 frames per episode) gave the same single equilibrium for
all five DQN seeds, including the four that did not learn.

The paper reported (0, 10) and (7, 3) as equilibria at 40 × 40. **That result
did not replicate here.** Differences that could explain it:

- caught prey respawn at a random spot, not at the densest spot (KDE) as in
  the paper, which keeps TurnAway prey out of the swarm the predator hunts;
- the predator's target radius, hold time and leap schedule are guesses;
  with random choice among nearby prey, prey that run away are rarely
  nearby, so SELFish prey that stay close act as shields;
- the learned SELFish policy is weaker than TurnAway even in homogeneous
  populations (2,552 vs 5,499 frames), which the paper's Nash results do
  not appear to share.

None of these was tested separately, so the cause is open.

Reference: [arXiv:1905.04077](https://arxiv.org/abs/1905.04077).
