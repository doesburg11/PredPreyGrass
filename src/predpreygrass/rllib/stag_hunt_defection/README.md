# Stag Hunt Defection (Join-or-Free-Ride)

This module is a full copy of `stag_hunt` with a minimal change that introduces
true defection. The ecology stays intact; only the cooperative hunting decision
is made voluntary at capture time. Predators can now free-ride on others who join
and pay a cost.

The defection module is non-vectorized only. The original `stag_hunt` and
`stag_hunt_vectorized` remain unchanged.

## What changed (high level)

- Predators have a second action component `join_hunt` that controls whether they
  contribute to a team capture.
- Capture uses only joiners, not all nearby predators.
- Joiners pay a fixed energy cost on successful capture.
- Non-joiners can receive a small scavenger spillover.
- Everything else stays the same: movement, energy decay, reproduction, LOS,
  grass growth, walls, and the original team-capture margin/split logic.

## Files and structure

This directory mirrors `stag_hunt` in full. Key files:

- `predpreygrass_rllib_env.py`: defection-enabled environment.
- `config/config_env_stag_hunt.py`: default env config with defection knobs.
- `tune_ppo.py`, `tune_ppo_resume.py`: training entry points wired to this module.
- `random_policy.py`: quick rollout viewer (predators now sample MultiDiscrete).
- `utils/*`: copied helpers (renderer, callbacks, scenario inspector, etc).

All imports inside this module point to `predpreygrass.rllib.stag_hunt_defection.*`
so you can run it without touching the original module. Default `ray_results`
paths also point at `.../stag_hunt_defection/ray_results/`.

## New action semantics

Only predators get the new join action. Prey actions are unchanged.

- Predator action space: `MultiDiscrete([move, join_hunt])`
  - `move`: same index as before (based on `type_*_action_range`).
  - `join_hunt`: `0` = refuse, `1` = join.
- Prey action space: `Discrete(move)` as before.

The env is backward compatible with old int actions for predators:

- If a predator action is a single int, it is treated as `join_hunt = 1`.
- If a predator action is a tuple/list/np array, it is interpreted as
  `[move, join_hunt]`.
- If a predator action is a dict, it looks for `move` and `join_hunt` keys.

Example action dict (manual stepping):

```
actions = {
    "type_1_predator_0": [4, 1],  # move index 4, join
    "type_1_predator_1": [4, 0],  # move index 4, free-ride
    "type_1_prey_0": 7,
}
```

## Join-or-Free-Ride capture logic

Let:
- `J` = predators in Moore neighborhood (Chebyshev <= 1) with `join_hunt = 1`
- `F` = nearby predators with `join_hunt = 0`
- `E` = prey energy
- `margin` = `team_capture_margin`
- `c_join` = `team_capture_join_cost`
- `s` = `team_capture_scavenger_fraction`

### Eligibility
Only predators in Moore neighborhood are eligible. If no joiners are present,
no capture attempt happens.

### Capture condition
Capture succeeds if:

```
sum(energy of J) > E + margin
```

Non-joiners do not count toward success.

### Success payouts
- If `F` is non-empty, a scavenger pool is reserved: `E * s`.
- If `F` is empty, the scavenger pool is zero.
- Joiners split the remaining energy (`E - scavenger_pool`) either equally or
  proportionally (based on `team_capture_equal_split`).
- Each joiner pays `c_join` after receiving its share.
- Each free rider gets `scavenger_pool / |F|`.

### Failure handling
- Failure penalties apply only to joiners.
- The existing penalty `energy_percentage_loss_per_failed_attacked_prey` is
  applied to joiners as before.
- `failed_attack_kills_predator` also applies only to joiners.
- Free riders never pay failure costs.

### Death from join cost
If a joiner drops to `<= 0` energy after paying the join cost, it dies with
cause `exhausted_hunt`.

## New config keys

Defined in `config/config_env_stag_hunt.py`:

- `team_capture_join_cost` (float): fixed energy cost paid by joiners on success.
- `team_capture_scavenger_fraction` (float in [0, 1]): fraction of prey energy
  reserved for nearby non-joiners (only when non-joiners are present).

These are in addition to existing team-capture controls:

- `team_capture_margin`
- `team_capture_equal_split`

## Info fields and event logs

Per-agent info additions (predators):

- `team_capture_free_riders`: number of non-joiners near the capture.
- `team_capture_scavenger_gain`: energy gained via scavenging (free riders only).
- `team_capture_join_cost`: join cost paid (joiners only).
- `team_capture_joined`: `True` if agent joined, else `False`.

Global info additions:

- `team_capture_last_free_riders` in `infos["__all__"]`.

Event log additions (predator eating/failed events):

- `free_riders`: list of free-riding predators nearby.
- `join_hunt`: boolean per event.
- `join_cost`: join cost applied (0 for free riders).

## Training and evaluation

Typical entry points:

- Train: `python src/predpreygrass/rllib/stag_hunt_defection/tune_ppo.py`
- Resume: `python src/predpreygrass/rllib/stag_hunt_defection/tune_ppo_resume.py`
- Random rollouts: `python src/predpreygrass/rllib/stag_hunt_defection/random_policy.py`
- Debug/eval: `evaluate_ppo_from_checkpoint_debug.py`
- Multi-run eval: `evaluate_ppo_from_checkpoint_multi_runs.py`

All scripts in this module are already wired to the defection env and config.

## Compatibility notes

- Predators now emit MultiDiscrete actions. Custom policies must output
  `[move, join_hunt]` for predator agents.
- Existing policies that output a single int will still work but always join.
- Prey behavior and action spaces are unchanged.
- Network helpers in `utils/networks.py` use `act_space.n` when present; for
  MultiDiscrete predators, `act_space.n` is not set, so it falls back to the
  standard FC head size. This does not block training.

## Design summary

This change adds a minimal social dilemma without adding communication,
reputation, or centralized logic. Predators can now choose whether to contribute
at the moment of capture, making free-riding a real, learnable strategy while
preserving the original ecological dynamics.
