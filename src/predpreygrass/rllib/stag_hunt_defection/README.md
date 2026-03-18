# Stag Hunt Defection (Join-or-Free-Ride)

<p align="center">
    <b>Emerging human cooperation, defection and free-riding</b></p>
<p align="center">
    <img align="center" src="./../../../../assets/images/gifs/stag_hunt_defect.gif" width="600" height="500" />
</p>

This module is a full copy of `stag_hunt` with an addition that introduces
defection. The ecology stays intact; only the cooperative hunting decision
is made voluntary at capture time. Predators can now free-ride on others who join
and pay a cost.

## What changed (high level)

- Predators have a second action component `join_hunt` that controls whether they
  contribute to a team capture.
- Capture uses only joiners, not all nearby predators.
- Joiners pay a fixed energy cost on successful capture.
- Defectors (non-joiners) can receive a small scavenger spillover.
- Everything else stays the same: movement, energy decay, reproduction, LOS,
  grass growth, walls, and the original team-capture margin/split logic.

## Files and structure

This directory `stag_hunt_defection` mirrors `stag_hunt`. Key files:

- `predpreygrass_rllib_env.py`: defection-enabled environment.
- `config/config_env_stag_hunt_defection.py`: default env config with defection knobs.
- `random_policy.py`: quick rollout viewer (predators now random sample MultiDiscrete).
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

Defectors (non-joiners) do not count toward success.

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
- Free riders never pay failure costs.

### Why defectors cannot solo-capture
`join_hunt = 0` means “refuse to contribute.” If defectors could still capture
alone, defection would become a safe default (no cost, still gets prey), which
weakens the social dilemma. Requiring `join_hunt = 1` for any capture keeps the
choice meaningful: cooperation enables success, defection can only free-ride.
This preserves the minimal-change goal and avoids a degenerate policy like
“defect unless others are present.”

### Death from join cost
If a joiner drops to `<= 0` energy after paying the join cost, it dies with
cause `exhausted_hunt`.

## Detailed change table

| Change area | Original stag_hunt | Defection version | Configure/observe |
| --- | --- | --- | --- |
| Predator action space | `Discrete(move)` | `MultiDiscrete([move, join_hunt])` with `join_hunt` in `{0,1}` | `predpreygrass_rllib_env.py`, per-step `join_hunt` |
| Capture eligibility | All nearby predators can contribute | Only joiners contribute; free riders do not count | `join_hunt` per step |
| Capture condition | Sum(nearby energies) > `E + margin` | Sum(joiner energies) > `E + margin` | `team_capture_margin` |
| Reward split | All helpers split full prey energy | Joiners split `E - scavenger_pool`; free riders equally share `scavenger_pool` | `team_capture_scavenger_fraction`, `team_capture_equal_split` |
| Cooperation cost | None on success | Joiners pay fixed `team_capture_join_cost` | `team_capture_join_cost`, event `join_cost` |
| Failure penalties | Apply to all helpers | Apply only to joiners | `energy_percentage_loss_per_failed_attacked_prey` |
| Defection metrics | Not defined | Join/defect rates and free-rider exposure tracked | `utils/defection_metrics.py`, `EpisodeReturn` |

Explanation notes:
- "Nearby" means Moore neighborhood (Chebyshev distance <= 1), same as the base env.
- "Joiners" are predators with `join_hunt = 1` on that step; "free riders" are `join_hunt = 0`.
- A scavenger pool exists only if at least one free rider is present; otherwise joiners split full prey energy.
- "Solo capture" means exactly one joiner; "coop capture" means two or more joiners.
- Join/defect rates are per predator-step; capture metrics use successful capture events only.

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
- `join_hunt`: per-step join choice for all predators (logged every step).

Global info additions:

- `team_capture_last_free_riders` in `infos["__all__"]`.

Event log additions (predator eating/failed events):

- `free_riders`: list of free-riding predators nearby.
- `join_hunt`: boolean per event.
- `join_cost`: join cost applied (0 for free riders).

Per-step agent data additions:

- `per_step_agent_data[step][predator_id]["join_hunt"]` records the join choice
  for every predator each step, independent of capture outcomes.

## Measuring defection/cooperation/solo

Use the helper script in `utils/defection_metrics.py` to summarize metrics from a
short rollout (defaults come from `config_env_stag_hunt.py`):

```bash
PYTHONPATH=src python -m predpreygrass.rllib.stag_hunt_defection.utils.defection_metrics
```

It reports:

- Join vs defect rates per predator-step.
- Solo vs cooperative capture rates (successful captures only).
- Free-rider exposure on successful captures.

RLlib training runs (e.g., `tune_ppo.py`) already use the `EpisodeReturn`
callback, which now logs these defection/cooperation metrics to `custom_metrics`
so they appear in TensorBoard.

## Sample results (from last evaluation output)

These tables capture the exact metrics you shared from the latest
`evaluate_ppo_from_checkpoint_multi_runs.py` run (seeds 1-10). The aggregate
uses a `min_steps` filter of 500 (kept 4 of 10 runs).

## Opportunity-conditioned preference (mammoth vs rabbit)

To avoid comparing this spatial ecology to a one-shot stag-hunt game, we use an
opportunity-conditioned preference test: only predator-steps where a prey is
adjacent (Moore neighborhood) are counted. The key question is whether
predators choose to `join_hunt` more often when mammoths are available than when
only rabbits are available.

From the latest eval folder:
`eval_checkpoint_000049_2026-01-06_23-24-46`

- Any prey available: join rate 0.844 (4446 / 5269)
- Mammoth available: join rate 0.848 (4305 / 5079)
- Rabbit available: join rate 0.749 (170 / 227)
- Mammoth only: join rate 0.848 (4276 / 5042)
- Rabbit only: join rate 0.742 (141 / 190)
- Both available: join rate 0.784 (29 / 37)

Interpretation:
- Joining is consistently higher when mammoths are present (0.848) than when
  only rabbits are present (0.742).
- This indicates a revealed preference to cooperate for the higher-risk,
  higher-return prey, even though rabbit captures are easier.
- The “both available” bucket is small but still shows a high join rate,
  suggesting predators do not simply default to solo rabbit hunting when a
  mammoth is in reach.

This is a more “pure” stag-hunt indicator because it conditions on local
opportunity rather than global capture counts, and it reflects the actual
join/defect choice made at the moment of potential cooperation.

### Attempt-based preference (unique attempts)

From the same eval folder (unique attempts grouped by `(t, prey_id)`):

- Mammoth attempts: 3509 (87.1% of all attempts)
- Rabbit attempts: 521 (12.9% of all attempts)
- Mammoth share of cooperative attempts: 96.9%
- Rabbit share of cooperative attempts: 3.1%

Risk/return profile:

- Mammoth success rate: 8.0% overall (coop 22.1%, solo 0.48%)
- Rabbit success rate: 78.7% overall (coop 97.4%, solo 77.2%)
- Energy per success: mammoth 13.07 vs rabbit 1.57
- Joiner net gain per success: mammoth 12.27 vs rabbit 1.35

Interpretation:
- Predators attempt mammoths far more often, and almost all cooperative attempts
  target mammoths.
- Mammoths are high risk but high return; rabbits are low risk and low return.
- Combined with the opportunity-conditioned join rates above, this supports a
  preference for the high-risk/high-return cooperative option when it is
  available.

### Aggregate (kept runs only; steps >= 500)

| Steps | Join steps | Defect steps | Total predator steps | Join rate | Defect rate | Captures | Solo | Coop | Solo rate | Coop rate | Joiners total | Free riders total | Free rider rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4000 | 150878 | 32016 | 182894 | 82% | 18% | 1809 | 471 | 1338 | 26% | 74% | 4929 | 190 | 4% |

### Per-run detailed table (all runs; kept if steps >= 500)

| Run | Steps | Kept | Join steps | Defect steps | Total predator steps | Join rate | Defect rate | Captures | Solo | Coop | Solo rate | Coop rate | Joiners total | Free riders total | Free rider rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1000 | yes | 37296 | 7716 | 45012 | 83% | 17% | 383 | 56 | 327 | 15% | 85% | 1128 | 54 | 5% |
| 2 | 206 | no | 7711 | 1937 | 9648 | 80% | 20% | 270 | 204 | 66 | 76% | 24% | 380 | 12 | 3% |
| 3 | 1000 | yes | 37491 | 7782 | 45273 | 83% | 17% | 401 | 73 | 328 | 18% | 82% | 1230 | 45 | 4% |
| 4 | 286 | no | 8745 | 2271 | 11016 | 79% | 21% | 286 | 228 | 58 | 80% | 20% | 387 | 9 | 2% |
| 5 | 324 | no | 9873 | 2949 | 12822 | 77% | 23% | 336 | 268 | 68 | 80% | 20% | 443 | 9 | 2% |
| 6 | 1000 | yes | 39368 | 7806 | 47174 | 83% | 17% | 526 | 180 | 346 | 34% | 66% | 1352 | 43 | 3% |
| 7 | 1000 | yes | 36723 | 8712 | 45435 | 81% | 19% | 499 | 162 | 337 | 32% | 68% | 1219 | 48 | 4% |
| 8 | 299 | no | 11846 | 2576 | 14422 | 82% | 18% | 329 | 216 | 113 | 66% | 34% | 551 | 7 | 1% |
| 9 | 134 | no | 4171 | 1016 | 5187 | 80% | 20% | 231 | 189 | 42 | 82% | 18% | 299 | 4 | 1% |
| 10 | 229 | no | 6740 | 1971 | 8711 | 77% | 23% | 260 | 210 | 50 | 81% | 19% | 335 | 10 | 3% |

Notes:
- Join/defect rates are per predator-step.
- Solo/coop rates are over successful captures only.
- Free rider rate is `free_riders_total / (joiners_total + free_riders_total)`.
- Rates shown as whole percentages (rounded).

### Results summary and conclusion

Across these runs, predators join most of the time (~80% join vs ~20% defect),
showing that cooperation is common but defection is non-trivial. Successful
captures split between solo and cooperative outcomes depending on run length:
shorter runs skew heavily toward solo captures, while the longer (kept) runs
show a strong cooperative majority (~74% coop). Free-rider exposure on successful
captures is consistently low (about 1% to 5%), which indicates that free riding
occurs but does not dominate capture outcomes in this configuration. Overall,
the mechanism achieves its goal: it introduces measurable defection behavior
without collapsing cooperation, yielding a stable mix of join and defect choices
that can be probed by norms or punishment in later experiments.

## Training and evaluation

Typical entry points:

- Train: `python src/predpreygrass/rllib/stag_hunt_defection/tune_ppo.py`
- Random rollouts: `python src/predpreygrass/rllib/stag_hunt_defection/random_policy.py`
- Debug/eval: `evaluate_ppo_from_checkpoint_debug.py`
- Multi-run eval: `evaluate_ppo_from_checkpoint_multi_runs.py`

All scripts in this module are already wired to the defection env and config.
Evaluation scripts decode MultiDiscrete actions, so checkpoints trained with
join/defect behavior are evaluated correctly.

## Quick Start

From the repo root:

```bash
# Train
PYTHONPATH=src python src/predpreygrass/rllib/stag_hunt_defection/tune_ppo.py

# Random rollout viewer
PYTHONPATH=src python src/predpreygrass/rllib/stag_hunt_defection/random_policy.py

# Evaluate a checkpoint (debug viewer)
PYTHONPATH=src python src/predpreygrass/rllib/stag_hunt_defection/evaluate_ppo_from_checkpoint_debug.py

# Evaluate multiple runs (batch)
PYTHONPATH=src python src/predpreygrass/rllib/stag_hunt_defection/evaluate_ppo_from_checkpoint_multi_runs.py
```

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
