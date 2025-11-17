# Kin Kickback Rewards in `rllib_termination`

This document describes the "kin kickback" reward mechanism added to the `PredPreyGrass` environment in the `rllib_termination` variant.

## Overview

The goal of kin kickbacks is to track and optionally reward parents based on the survival and activity of their offspring. This allows analysis of lineage success and, in future variants, could support experiments on kin-based altruism.

Key ideas:

- Each agent keeps a counter `kin_kickbacks` in its lifetime statistics.
- On each step, if a child is alive and its parent is also alive, the parent may receive a small reward increment (a "kickback").
- These kickbacks are accumulated into both the per-step reward stream and the parent’s `cumulative_reward`.
- At the end of an episode, `kin_kickbacks` is written into the agent’s final record and exported in evaluation summaries.

## Configuration

The environment reads kin kickback magnitudes from the config:

- `kin_kick_back_reward_predator`
- `kin_kick_back_reward_prey`

These are exposed as:

```python
self.kin_kick_back_predator_config = config["kin_kick_back_reward_predator"]
self.kin_kick_back_prey_config = config["kin_kick_back_reward_prey"]
```

They can be scalars or type-specific dicts, analogous to other reward dictionaries. A value of `0.0` disables kin kickback rewards (counters will still be tracked if the logic is active).

## Environment Logic

### Tracking parent–child relations

When a new agent is created via reproduction (predator or prey):

- The parent’s `agent_id` is recorded in `self.agent_parents[child_id]`.
- The parent’s offspring bookkeeping is updated:
  - `self.agent_offspring_counts[parent] += 1`
  - `self.agent_live_offspring_ids[parent].append(child_id)`
- The child is registered via `_register_new_agent`, which initializes:

```python
self.agent_stats_live[agent_id] = {
    ...
    "parent": parent_agent_id,
    "offspring_count": 0,
    "offspring_ids": self.agent_live_offspring_ids[agent_id],
    ...
    "cumulative_reward": 0.0,
    "kin_kickbacks": 0,
}
```

### Per-step kin survival reward

In `step`, after engagements and reproduction and after assembling default reward/termination/truncation dictionaries, the environment applies the kin survival reward:

- For each active agent `child` in `self.agents`:
  - Look up its parent: `parent = self.agent_parents.get(child)`.
  - If `parent is not None` and `parent` is still in the set of alive agents:
    - Add a kin reward to the parent’s per-step reward:
      - `self.rewards[parent] = self.rewards.get(parent, 0.0) + kin_reward`
    - Update the parent’s live record:
      - `parent_record["cumulative_reward"] += kin_reward`
      - `parent_record["kin_kickbacks"] = parent_record.get("kin_kickbacks", 0) + 1`

The actual `kin_reward` scalar can be chosen per species (e.g. via `kin_kick_back_reward_predator` / `kin_kick_back_reward_prey`) and is intended to be small relative to primary task rewards.

This logic means:

- `kin_kickbacks` counts **how many per-step kin rewards** a parent received during its lifetime.
- `cumulative_reward` includes both normal task rewards (eating, reproduction, step rewards) and kin survival rewards.

### Finalization and persistence

At agent death or episode truncation, `_finalize_agent_record` moves the live record into `agent_stats_completed` and ensures `kin_kickbacks` is preserved:

```python
record = self.agent_stats_live.pop(agent_id, None)
if record is None:
    return
...
record["kin_kickbacks"] = record.get("kin_kickbacks", 0)
self.agent_stats_completed[agent_id] = record
```

`get_all_agent_stats()` then exposes both live and completed records with `kin_kickbacks` included.

## Evaluation Outputs

The kin kickback statistics are surfaced in the evaluation/debug scripts under `rllib_termination`:

- `evaluate_ppo_from_checkpoint_debug.py` uses `env.get_all_agent_stats()` to build per-agent stats.
- For each agent, it records:
  - `cumulative_reward`
  - `lifetime`
  - `offspring_count`
  - `kin_kickbacks`
- These are written to:
  - `agent_fitness_stats.json` – detailed per-agent records.
  - `reward_summary.txt` – human-readable ranked fitness summary, including a `KinKick` column.

There is also a dedicated analysis script:

- `test/analyze_kin_rewards.py`
  - Loads `agent_fitness_stats.json`.
  - Groups agents by policy group (e.g. `type_1_predator`).
  - Computes correlations between `kin_kickbacks`, offspring counts, and cumulative rewards.
  - Compares agents with `kin_kickbacks > 0` vs `kin_kickbacks == 0`.

This allows you to quantify how strongly kin survival rewards and lineage success are aligned in a given experiment.

## Usage Notes and Experiments

- To **disable** kin rewards but still track lineage structure:
  - Set `kin_kick_back_reward_predator = 0.0` and `kin_kick_back_reward_prey = 0.0` in the environment config.
  - `kin_kickbacks` counters will still be updated by the step logic (if enabled), but they will contribute `0` to rewards.
- To **enable** kin survival reward shaping:
  - Choose small positive values (e.g. `0.01` or `0.1`) relative to main reproduction/foraging rewards to avoid dominating the learning signal.
- For evolutionary comparisons:
  - Run a baseline experiment with kin rewards disabled.
  - Run a matched experiment with the same config but non-zero kin rewards.
  - Compare the outputs of `analyze_kin_rewards.py` across runs.

## Limitations and Future Directions

- The current kin kickback mechanism rewards parents whenever children are alive, but it does not add new actions (e.g., energy transfer) or explicit kin markers in observations.
- As a result, it primarily reinforces lineages that are already successful rather than creating qualitatively new "altruistic" behaviors by itself.
- Future extensions may add:
  - Explicit energy-sharing actions between kin.
  - Limited per-step resource intake and carcass mechanics to create sharing opportunities.
  - Additional shaping terms tied to offspring survival milestones or descendants’ reproduction events.

This file should be kept in sync with any changes to the kin reward logic in `predpreygrass_rllib_env.py` and the related evaluation scripts.
