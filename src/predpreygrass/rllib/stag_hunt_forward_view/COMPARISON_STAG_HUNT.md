# Stag Hunt Defection vs Forward-View Evaluation: Qualitative Comparison (1000-step runs)

This note compares `stag_hunt_defection` with `stag_hunt_forward_view` using the
1000-step evaluation plots currently available in this repo. It is a qualitative,
plot-based analysis (not a full statistical study).

## Inputs used

- Defection env: `src/predpreygrass/rllib/stag_hunt_defection/predpreygrass_rllib_env.py`
- Defection config: `src/predpreygrass/rllib/stag_hunt_defection/config/config_env_stag_hunt_defection.py`
- Defection eval plots (1000 steps):
  - `src/predpreygrass/rllib/stag_hunt_defection/ray_results/.../summary_plots/iters=1000`

- Forward-view env: `src/predpreygrass/rllib/stag_hunt_forward_view/predpreygrass_rllib_env.py`
- Forward-view config: `src/predpreygrass/rllib/stag_hunt_forward_view/config/config_env_stag_hunt_forward_view.py`
- Forward-view eval plots (1000 steps, join_cost_0.02, rabbits removed):
  - `src/predpreygrass/rllib/stag_hunt_forward_view/ray_results/join_cost_0.02/STAG_HUNT_FORWARD_VIEW_JOIN_COST_0_02_SCAVENGER_0_3_2026-01-29_15-52-24/PPO_PredPreyGrass_1fe3e_00000_0_2026-01-29_15-52-25/checkpoint_000099/eval_10_runs_STAG_HUNT_FORWARD_VIEW_2026-02-04_23-06-13/visuals/evolution_summary_*.png`

Legend reference (per `stag_hunt/README.md`):
- `type_1_prey` = mammoths
- `type_2_prey` = rabbits

## Qualitative findings from the plots

### `stag_hunt_defection`

- **Rabbits crash early:** `type_2_prey` spikes in the first ~100 steps and then
  collapses to (near) zero for the remainder of each run, indicating rapid rabbit
  extinction in these defection-only runs.
- **Mammoths persist at moderate levels:** `type_1_prey` stabilizes around
  mid‑range counts (roughly 20–40) with continued oscillations rather than extinction.
- **Predators remain high and oscillatory:** `type_1_predator` rises quickly and
  then fluctuates around ~40–70, showing predator–prey cycles without long-term collapse.
- **Single predator type:** `type_2_predator` stays at zero throughout.

### `stag_hunt_forward_view` (join_cost_0.02, rabbits removed)

- **Only one prey type present:** These evals removed rabbits, so only `type_1_prey`
  appears. This makes prey‑type comparisons impossible in this set.
- **Strong early boom–bust:** Predators spike well above 100 early, then crash,
  followed by repeated oscillations for the remainder of the 1000 steps.
- **Prey recover and persist:** `type_1_prey` drops sharply early, then rebounds
  into a stable oscillatory band (roughly 35–60) without going extinct.
- **Longer oscillatory cycles:** The predator–prey dynamics show pronounced,
  repeating cycles rather than settling to a flat equilibrium.

## Interpretation (forward-view effect)

### 1) Stag hunt principle

In the base environment, capture is effectively always cooperative because all
nearby predators contribute. That makes mammoth hunting relatively reliable and
can drive mammoths to extinction early. This shifts the ecology toward rabbits
(the lower-energy prey) as the stable option.

These two plot sets are **not perfectly matched**: the forward‑view evals used
`join_cost_0.02` with rabbits removed, while the defection-only plots include both
prey types. So differences here could be driven by **prey composition**, **cost
settings**, **forward-view observation**, or all three. Treat comparisons as
qualitative signals, not causal evidence.

### 2) Cooperation vs defection

Defection does not eliminate cooperation (join rates were still around 80% in
the sample metrics), but forward-view may change which prey are encountered and
how quickly partners align. This can shift prey persistence even when social
incentives are unchanged.

### 3) Additional observations

- `type_2_predator` is flat at zero in both configurations, so the system is
  effectively single-predator-type vs two prey types.
- The base env shows higher variance across runs (one run flips the outcome),
  suggesting more sensitivity to stochasticity or local conditions.

## Caveats

- Defection side is four 1000‑step runs from `summary_plots/iters=1000`.
- Forward‑view side is a 10‑run evaluation set (individual `evolution_summary_*.png`),
  but with rabbits removed and join costs active.
- To validate the pattern, consider generating **matched** defection-only forward‑view
  plots (same prey mix, same costs) and aggregating time‑to‑extinction/mean populations.

## Summary conclusion

The plots suggest that **defection-only** runs quickly eliminate rabbits and settle
into a mammoth–predator oscillation, while the **forward‑view join_cost** runs (with
rabbits removed) show strong early predator overshoot and sustained oscillations
without extinction. Because these settings are not matched, this is **not** evidence
of a clean forward‑view effect. A clean conclusion requires matched defection-only
forward‑view plots with the same prey mix and costs.
