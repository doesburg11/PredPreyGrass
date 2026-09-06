# Predator-Prey-Grass drive-conditioned environment

This environment starts as a copy of [`base_environment`](../base_environment). The current implementation is intentionally still close to that baseline so drive-conditioned behavior can be added and reviewed incrementally.

## Purpose

The core idea: make hidden, sample-expensive-to-discover internal state directly visible to the agent, without touching the reward signal or telling the policy what to do about it.

**The problem it's addressing**: in the plain baseline, an agent only sees raw local grid channels (predator/prey/grass energy density in its window) — its own energy is tracked internally but never directly exposed as a feature. To act well, PPO has to *implicitly* learn things like "my energy is low → that predicts starvation → I should prioritize foraging" or "my energy just crossed some threshold → reproduction is now possible" purely from correlating raw pixel-like inputs with returns over many samples. That correlation is learnable, but expensive — the biologically meaningful abstraction has to be re-derived by the network every time; it's not literally present in the input.

**What drive conditioning does**: since the environment has privileged, exact access to each agent's own energy and to the local densities it already computes for the raw channels, it directly computes a handful of biologically-interpretable pressure signals (`hunger_pressure`, `reproductive_readiness`, `prey_opportunity`, `predator_danger_pressure`, `grass_opportunity`) and hands them to the agent as extra observation channels — broadcasting each scalar as a constant value across the local window, so it slots into the existing CNN-friendly `C×H×W` tensor with no architecture changes needed.

**The important restraint, which is the whole design point**: it stops at *description*, not *prescription*. `hunger_pressure` tells the agent "your situation is getting dangerous" — it doesn't say which direction to move, whether to flee or forage, or which prey to chase. Contrast that with rejected features like `best_escape_direction` or `can_kill_this_prey` (see "Rationale" below) — those would hand the policy a tactical answer, defeating the point of using RL to *discover* behavior. Reward stays reproduction-only, actions stay movement-only — only the *input representation* changes.

**How this differs from `project_reward_shaping`**: that investigation found that adding *density* to the **reward channel** (a continuous per-step signal layered onto the sparse reproduction bonus) actively hurt learning — it added noise that made credit assignment harder, even though the extra signal was informative in principle. Drive-conditioning is a structurally different move: it enriches the **observation channel**, not the reward channel. It's not subject to that same failure mode, since it never touches what the agent is rewarded for — only what the agent can see going into its own policy network. That's why this module is a separate, standalone experiment rather than a `project_reward_shaping` variant (see below).

## Current baseline

- Predators, prey, and grass are randomly placed in a gridworld at reset.
- Predators and prey are learning agents with separate RLlib policies.
- Grass is a non-learning environment resource.
- Agents observe only a local window around their position.
- Movement costs energy every step.
- Prey gain energy by eating grass.
- Predators gain energy by catching prey.
- Predators and prey reproduce asexually once their energy crosses the configured threshold.
- New agents spawn near their parent.
- Rewards are sparse by default: reproduction is rewarded, while eating, catching, step, and death rewards can be configured in [`config_env.py`](./config_env.py).
- Training uses [`tune_ppo_drive_conditioned_environment.py`](./tune_ppo_drive_conditioned_environment.py).
- Interactive evaluation uses [`evaluate_ppo_from_checkpoint_debug.py`](./evaluate_ppo_from_checkpoint_debug.py).

## Status

Drive-conditioned logic is implemented on top of the copied baseline (the "conservative first feature set" described below), while the original `base_environment` remains unchanged. Verified correct: observation channel counts match expectations (7 for predators, 8 for prey, world + drive channels), and all 5 drive-feature formulas were verified numerically at both boundary and mid-range values. A first baseline-vs-drive-conditioned (full drive set) comparison has now been run — see **Results** below. Not yet done: the energy-only arm, additional seeds, and the stronger-affordances arm of the originally planned three-way comparison.

## Results: baseline vs. drive-conditioned, single seed (2026-09-06)

**Key message**: adding drive-conditioned observations made agents reproduce about 6% more successfully overall — mostly by helping predators specifically — in this one training run. A promising early sign, not proof yet, since it's only been tested once.

**Setup**: `base_environment` and `drive_conditioned_environment --drive-set full`, both seed 42, both 1000 PPO training iterations, identical `config_env.py` settings and PPO hyperparameters (verified like-for-like beforehand — the only difference is the 3-4 extra drive channels drive-conditioned adds to the observation). Runs: `PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45` and `PPO_DRIVE_CONDITIONED_FULL_SEED42_2026-09-06_07-26-38` under `~/simulation_results/ray_results/`. Both completed cleanly (1000/1000 iterations, no crashes, no extinctions to speak of — `extinct_predator`/`extinct_prey` ≈ 0 throughout).

**Headline**: drive-conditioned showed a **+6.0% higher pooled return** over the full run — 7322.6 vs. 6905.1, episode-count-weighted across 1057 vs. 1055 completed episodes (pooling by actual completed episodes, not by per-iteration means, since many iterations complete only 0-3 episodes and a naive per-iteration-mean read is noisy/misleading — an extraction bug an independent Codex review caught and corrected in this analysis).

**Shape of the effect over training** (episode-pooled return, gap vs. baseline):

| Iterations | Gap |
|---|---|
| 1–25 (both still learning to survive) | +76.0% |
| 26–50 (both just reached full-length episodes) | +28.9% |
| 51–100 | +9.3% |
| 101–200 | +4.4% |
| 201–300 | +6.0% |
| 301–400 | +6.5% |
| 401–500 | +6.6% |
| 501–600 | +3.1% |
| 601–700 | +2.8% |
| 701–800 | +5.3% |
| 801–900 | +5.3% |
| 901–1000 | +7.3% |

A large transient advantage during the early survival-learning phase, then a positive-but-variable advantage (roughly +3% to +9%) that persists to the end of training without decaying to zero. This early-transient-then-persistent shape is consistent with the variance-reduction argument for why drive-conditioning should help (see "Conceptual evidence" below) — though see the caveat at the end of this section before reading too much into it.

**Mechanism**: reward is reproduction-only (10.0 per birth for either species), so return is an exact identity: `return = 10 × (predator_births + prey_births)`. The advantage is disproportionately a **predator** effect — predator births explain 70.7% of the full-run return gap, and essentially all (~100%) of the gap in the back half of training (iterations 501-1000). Prey births were only modestly higher for drive-conditioned (+1-2%) throughout.

**A population-composition effect seen mid-run did not hold up**: around iteration 332, drive-conditioned showed a notably more prey-heavy population (26.3 prey/17.3 predators vs. baseline's 14.0/22.0). Checked against the completed run, this was transient — pooled over iterations 501-1000, final population composition is essentially identical between arms (~17-18 predators, ~23-24 prey either way). Flagged here specifically because it looked real at the time and would have been a misleading claim if reported without checking the full trajectory.

**Drive-channel calibration held up**: none of the five drive channels saturated near a constant value during this run — each showed a real spread from ~0 to ~1 with a non-degenerate mean (`hunger_pressure` mean 0.15, `reproductive_readiness` 0.48, `prey_opportunity` 0.50, `predator_danger_pressure` 0.55, `grass_opportunity` 0.39). This resolves the "unvalidated normalizer constants" caveat from earlier in this document, at least for this seed.

**The caveat that matters most**: this is a single seed per arm. Per an independent Codex review of the raw data: *"state it as a documented single-run observation, not evidence of a statistically robust algorithmic advantage without multi-seed replication and variance/confidence intervals."* The consistent positive sign across nearly every 100-iteration window in this one run is suggestive, but a different seed could plausibly show a smaller, larger, or reversed gap purely from randomness.

### TODO: energy-only arm (not yet run)

The two-group prediction earlier in this document (see "Rationale") is that `hunger_pressure`/`reproductive_readiness` — own-energy-based, encoding thresholds the raw observation cannot otherwise contain — should matter more than the three local-density-based drives (`prey_opportunity`, `predator_danger_pressure`, `grass_opportunity`), which a CNN could plausibly learn to approximate on its own from the raw channels. The completed comparison above found the advantage was disproportionately a **predator** effect (predator births explain ~71% of the full-run gap, ~100% of the back-half gap) — worth knowing whether that's coming from the 2 energy-based channels alone or needs the density-based ones too.

`--drive-set energy_only` (predator + prey each get just `hunger_pressure`/`reproductive_readiness`, no density channels) is already implemented and CLI-ready specifically to test this. A run was started on 2026-09-06 (seed 42, same setup as above) but was killed shortly after launch and its output removed — it was launched as an unplanned add-on immediately after the base-vs-drive result, without re-confirming the ~13-hour GPU cost with the user first, and was stopped once that was clarified. No data exists from it; this is a clean deferral, not an interrupted or failed run.

**To pick this back up**: `python -m predpreygrass.non_evolutionary.drive_conditioned_environment.tune_ppo_drive_conditioned_environment --seed 42 --drive-set energy_only --max-iters 1000`, then compare against the `full` run above the same way (episode-count-weighted pooling, not per-iteration means — see "Results" above for why). If `energy_only` captures most of the +6% gap, that confirms the two-group prediction; if it captures little of it, the density-based drives (or their interaction with the energy-based ones) are doing more of the work than predicted. Additional seeds (for both `full` and `energy_only`) remain the other open item — this single-seed result is a documented observation, not yet a validated effect.

## Expected advantages (predictions for future work, partially checked above)

Two separate questions worth keeping apart when the baseline-vs-drive-conditioned comparison finally runs: does it train faster, and is it worth having regardless of speed?

**Will it speed up training? Plausibly, but unevenly across the 5 features** — they split into two groups with different expected effect sizes:

1. **`hunger_pressure` and `reproductive_readiness`** (own-energy-based) are the more likely source of a real speedup. They encode information the raw observation *cannot actually contain at all*: the env-side constants `predator_hunger_safe_energy`/`prey_hunger_safe_energy` and the reproduction-energy thresholds. The raw observation only has the agent's own energy sitting at the center pixel of its own density channel — the network has to (a) learn to specifically attend to that one pixel amid an otherwise-irrelevant density map, and (b) learn the correct nonlinear rescaling against a threshold it can never directly observe, purely by correlating outcomes with reward over many samples. Handing over the pre-normalized `[0,1]` value skips both learning problems.
2. **`prey_opportunity`, `predator_danger_pressure`, `grass_opportunity`** (local density sums) are a weaker case for a speed benefit. Each is literally `np.sum(observation[channel])` over a channel that's already fully present in the raw input — close to the easiest operation a CNN can learn (a 1x1 all-ones conv plus pooling), so a randomly-initialized network is likely to pick this up reasonably fast on its own. The gain from pre-computing it is probably real (removes some early-training variance) but smaller than for the energy-based drives.

**Advantages independent of raw training speed** — worth weighting as importantly as the speed question when evaluating results:

- **Interpretability**: lets you read off what the agent "believes" its hunger/danger/opportunity level is at any timestep and correlate it with behavior, instead of probing a black-box CNN's internal activations.
- **Ablation-friendly by design**: `enable_drive_channels` and the per-species drive-channel lists are already config-toggleable, enabling controlled "which specific drive matters" experiments that are much harder to run against an implicit, emergent representation.
- **Possible generalization benefit** (speculative): a policy conditioned on a normalized `[0,1]` drive signal may transfer better across different hyperparameter settings (grid size, initial energy, thresholds) than one that learned to read raw, un-normalized pixel values tied to the specific numbers seen during training.
- **Value-function variance reduction**: PPO's value estimates may stabilize faster with an explicit "urgency" signal available immediately, rather than the critic having to slowly discover the correlation between raw pixel patterns and eventual returns — a plausible, if hard-to-isolate, contributor to sample efficiency in sparse-reward settings.

**Caveat**: none of the above is measured yet. The normalizer constants (`prey_opportunity_normalizer`, `predator_danger_normalizer`, `grass_opportunity_normalizer`) are unvalidated design choices — if poorly calibrated for actual local densities, the derived features could end up near-constant or noisy rather than informative. The three-way comparison below is what would actually confirm or refute any of these predictions, including whether the two feature groups really do split in effect size the way predicted here.

## Rationale

Yes, this can probably be implemented more efficiently with derived drive features without steering the agents too directly toward a hand-coded goal.

The important design boundary is:

```text
Keep:
    reward = reproduction only
    action space = movement only

Add:
    internal-state and ecological-context signals in the observation
```

The current sparse-reward setup is open-ended, but expensive. Agents have to discover through trial and error that low energy predicts starvation, high energy enables reproduction, nearby enemies can be dangerous, nearby allies can matter, and grass density affects future survival. PPO can learn this, but it costs a lot of samples because much of the biological meaning is implicit.

Drive-conditioned observations make the state more legible without giving the policy the answer. A feature such as `hunger_pressure` does not say which direction to move. It only tells the agent that its internal state is becoming dangerous. The learned policy must still discover whether that should lead to foraging, fleeing, clustering, risk-taking, or waiting.

A conservative first feature set is:

```text
Predator:
    hunger_pressure
    reproductive_readiness
    prey_opportunity

Prey:
    hunger_pressure
    reproductive_readiness
    predator_danger_pressure
    grass_opportunity
```

These are biologically plausible motivational or local-resource signals rather than engineered tactical advice. `isolation_pressure` is intentionally left out of this first version because this environment is not yet trying to study emerging cooperation. Predator `danger_pressure` is also left out because, in this baseline ecology, predators mainly die from starvation rather than from direct predation or combat.

The implemented channels are different from stronger affordance features such as:

```text
best_grass_direction
best_escape_direction
can_kill_this_prey
best_hunt_target
```

Those later features may improve learning, but they inject more designer assumptions. The drive-conditioned version should start with non-directional scalar drives and only add explicit affordances if the experiment shows they are needed.

For the current CNN-style observation, the simplest implementation is to broadcast each scalar drive as an extra constant channel over the local observation window:

```text
old observation:
    C x H x W

new observation:
    (C + drive_channels) x H x W
```

This keeps the environment compatible with image-like RLlib observations while adding a small motivational layer. In nature/nurture terms, PPO still learns the movement behavior during training, while the environment supplies a more biologically plausible internal state interface.

The experiment should compare:

```text
1. Baseline:
   local grid + raw energy

2. Drive-conditioned:
   local grid + raw energy + asymmetric hunger/reproduction/food/danger drives

3. Optional later:
   add stronger ecological affordances only if needed
```

Useful metrics are sample efficiency, episode length, extinction timing, birth and death rates, predator/prey population stability, and whether both species survive to the horizon more reliably.
