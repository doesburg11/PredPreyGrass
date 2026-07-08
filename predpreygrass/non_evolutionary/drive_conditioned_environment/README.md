# Predator-Prey-Grass drive-conditioned environment

This environment starts as a copy of [`base_environment`](../base_environment). The current implementation is intentionally still close to that baseline so drive-conditioned behavior can be added and reviewed incrementally.

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

## Next step

Add drive-conditioned logic on top of this copied baseline while keeping the original `base_environment` unchanged.

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
