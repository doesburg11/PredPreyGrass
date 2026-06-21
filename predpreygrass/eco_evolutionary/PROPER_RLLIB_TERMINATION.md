# RLlib Termination Semantics

This note documents the current `eco_evolutionary` agent-lifetime protocol.
It replaces older migration notes that described wall/occlusion mechanics and
type-specific agents that are no longer part of this module.

## RLlib Contract

RLlib `MultiAgentEnv` episodes return dictionaries from `reset()` and `step()`.
The observation dictionary controls which agents are expected to act on the next
environment step. Rewards, terminations, truncations, and infos may contain
agents beyond the next acting set, for example an agent that was terminated by
another agent's action.

The special `__all__` key in the termination or truncation dictionaries ends the
whole multi-agent environment episode. Individual agent IDs can terminate or
truncate earlier while the environment episode continues.

Reference:
https://docs.ray.io/en/latest/rllib/multi-agent-envs.html

## Biological Lifetimes

In this environment, an RLlib agent ID represents one biological individual for
one lifetime. When an individual dies because it is eaten, starves, or exceeds
an age limit, that agent ID is marked done with:

```python
terminations[agent_id] = True
truncations[agent_id] = False
```

When the entire environment reaches the configured time limit, still-active
agents are marked:

```python
terminations[agent_id] = False
truncations[agent_id] = True
truncations["__all__"] = True
```

When the ecosystem-level episode ends because predators or prey go extinct, the
remaining live agents are terminated and `terminations["__all__"]` ends the
environment episode.

## No Agent-ID Reuse

Historical PredPreyGrass variants reused fixed agent slots for multiple
biological individuals. That made RLlib see several biological lifetimes as one
continuous single-agent trajectory, which could carry rewards, hidden state, and
episode statistics across deaths and births.

`eco_evolutionary` avoids that by never reusing an agent ID within an
environment episode. New offspring are allocated fresh IDs from the configured
`possible_agents` pool:

```text
predator_0, predator_1, ...
prey_0, prey_1, ...
```

This keeps the RLlib trajectory boundary aligned with the biological lifetime
boundary. The tradeoff is that `n_possible_predators` and `n_possible_prey`
must be large enough for the total number of individuals that may exist during
one environment episode, not merely the maximum number alive at one time.

## Current Observation Model

The current environment is an open grid. It has no wall channel, no internal
walls, and no line-of-sight occlusion.

Observations have three channels:

```text
0: predators
1: prey
2: grass
```

Grid edges are handled by clipping the observation window to valid grid
coordinates. Cells outside the grid remain zero in the returned observation
tensor. Movement uses the same boundary convention: proposed moves are clipped
to stay inside the grid.

## Practical Invariants

The implementation should preserve these invariants:

- `possible_agents` contains every agent ID that can appear in an episode.
- `agents` contains only currently active, acting biological individuals.
- A dead individual is removed from `agents` after its terminal step.
- A new offspring gets a fresh ID that has not previously been used in the same
  environment episode.
- `__all__` is set only when the whole environment episode is done.
- Agent trajectory boundaries are biological lifetime boundaries.

These rules are the reason the module uses large never-reuse ID pools instead
of recycling fixed predator/prey slots.
