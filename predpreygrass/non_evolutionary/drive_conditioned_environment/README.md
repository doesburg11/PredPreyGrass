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

Drive-conditioned logic is now implemented on top of the copied baseline (the "conservative first feature set" described below), while the original `base_environment` remains unchanged. Verified correct: observation channel counts match expectations (7 for predators, 8 for prey, world + drive channels), and all 5 drive-feature formulas were verified numerically at both boundary and mid-range values. Not yet done: the three-way baseline-vs-drive-conditioned-vs-stronger-affordances comparison this module exists to run — no results doc exists yet.

## Expected advantages (predictions for future work, not yet validated)

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
