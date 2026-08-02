# Predator-Prey-Grass drive-conditioned environment

This environment starts as a copy of [`base_environment`](../base_environment). The current implementation is intentionally still close to that baseline so drive-conditioned behavior can be added and reviewed incrementally.

## Purpose

The core idea: make hidden, sample-expensive-to-discover internal state directly visible to the agent, without touching the reward signal or telling the policy what to do about it.

**The problem it's addressing**: in the plain baseline, an agent only sees raw local grid channels (predator/prey/grass energy density in its window) — its own energy is tracked internally but never directly exposed as a feature. To act well, PPO has to *implicitly* learn things like "my energy is low → that predicts starvation → I should prioritize foraging" or "my energy just crossed some threshold → reproduction is now possible" purely from correlating raw pixel-like inputs with returns over many samples. That correlation is learnable, but expensive — the biologically meaningful abstraction has to be re-derived by the network every time; it's not literally present in the input.

**What drive conditioning does**: since the environment has privileged, exact access to each agent's own energy and to the local densities it already computes for the raw channels, it directly computes a handful of biologically-interpretable pressure signals (`hunger_pressure`, `reproductive_readiness`, `prey_opportunity`, `predator_danger_pressure`, `grass_opportunity`) and hands them to the agent as extra observation channels — broadcasting each scalar as a constant value across the local window, so it slots into the existing CNN-friendly `C×H×W` tensor with no architecture changes needed.

**The important restraint, which is the whole design point**: it stops at *description*, not *prescription*. `hunger_pressure` tells the agent "your situation is getting dangerous" — it doesn't say which direction to move, whether to flee or forage, or which prey to chase. Contrast that with rejected features like `best_escape_direction` or `can_kill_this_prey` (see "Rationale" below) — those would hand the policy a tactical answer, defeating the point of using RL to *discover* behavior. Reward stays reproduction-only, actions stay movement-only — only the *input representation* changes.

**How this differs from `project_reward_shaping`**: that investigation found that adding *density* to the **reward channel** (a continuous per-step signal layered onto the sparse reproduction bonus) actively hurt learning — it added noise that made credit assignment harder, even though the extra signal was informative in principle. Drive-conditioning is a structurally different move: it enriches the **observation channel**, not the reward channel. It's not subject to that same failure mode, since it never touches what the agent is rewarded for — only what the agent can see going into its own policy network. That's why this module is a separate, standalone experiment rather than a `project_reward_shaping` variant (see below).

### RLlib-compliance fix

This environment previously had the same two RLlib-compliance bugs found and fixed in `base_environment` (unsurprising, since this module is a direct copy of it): a dying agent's terminal transition (`terminated=True`, final reward, final observation) was silently dropped before reaching RLlib, and newborn agent IDs were recycled within an episode in a way that could conflate two unrelated individuals' trajectories into one. Both are now fixed the same way — terminating agents stay listed through the step they die in, and newborn IDs are assigned from a monotonically increasing, never-reused counter (`n_possible_predators`/`n_possible_prey` raised to `2000` accordingly). Verified: RLlib pre-check passes, zero ID reuse across 3 seeds with 70 real deaths tracked correctly, observation channel counts match expectations (7 for predators, 8 for prey, world + drive channels), all 5 drive-feature formulas verified numerically at both boundary and mid-range values, and a 2-iteration PPO smoke-run completes with no RLlib hard-errors.

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

Drive-conditioned logic is now implemented on top of the copied baseline (the "conservative first feature set" described below), while the original `base_environment` remains unchanged. Verified correct (see "RLlib-compliance fix" above for the verification method). Not yet done: the three-way baseline-vs-drive-conditioned-vs-stronger-affordances comparison this module exists to run — no results doc exists yet.

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
