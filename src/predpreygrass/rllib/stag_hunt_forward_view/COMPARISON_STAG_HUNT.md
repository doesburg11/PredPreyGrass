# Stag Hunt Defection vs Forward-View Defection: Evaluation Comparison (1000-step runs)

This note compares `stag_hunt_defection` with `stag_hunt_forward_view`
(defection + forward-view predators) using the 1000-step evaluation plots you
provided. It is a qualitative, plot-based analysis (not a full statistical study).

## Inputs used

- Defection env: `src/predpreygrass/rllib/stag_hunt_defection/predpreygrass_rllib_env.py`
- Defection config: `src/predpreygrass/rllib/stag_hunt_defection/config/config_env_stag_hunt_defection.py`
- Defection eval plots (1000 steps):
  - `src/predpreygrass/rllib/stag_hunt_defection/ray_results/.../summary_plots/iters=1000`

- Forward-view defection env: `src/predpreygrass/rllib/stag_hunt_forward_view/predpreygrass_rllib_env.py`
- Forward-view defection config: `src/predpreygrass/rllib/stag_hunt_forward_view/config/config_env_stag_hunt_defection.py`
- Forward-view defection eval plots (1000 steps):
  - `src/predpreygrass/rllib/stag_hunt_forward_view/ray_results/STAG_HUNT_DEFECT_PRED_LOSS_0_08_2026-01-05_01-26-09/PPO_PredPreyGrass_225cc_00000_0_2026-01-05_01-26-10/checkpoint_000029/eval_multiple_runs_STAG_HUNT_DEFECT_2026-01-05_14-04-28/summary_plots/iters=1000`

Legend reference (per `stag_hunt/README.md`):
- `type_1_prey` = mammoths
- `type_2_prey` = rabbits

## Qualitative findings from the plots

### `stag_hunt_defection`

- Replace with the observed pattern from your defection-only plots.

### `stag_hunt_forward_view` (defection + forward view)

- Replace with the observed pattern from the forward-view defection plots.

## Interpretation (forward-view effect)

### 1) Stag hunt principle

In the base environment, capture is effectively always cooperative because all
nearby predators contribute. That makes mammoth hunting relatively reliable and
can drive mammoths to extinction early. This shifts the ecology toward rabbits
(the lower-energy prey) as the stable option.

Defection logic is the same in both variants. Any differences between these two
sets of plots should be attributed primarily to the forward-view observation change
(predators see more ahead and less behind), which can alter encounter rates and
coordination timing.

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

- This is based on four 1000-step runs per condition from the provided plots.
  It is a qualitative comparison, not a statistical test.
- To validate the pattern, consider aggregating time-to-extinction and mean
  population levels across many runs.

## Summary conclusion

The plots should be read as a comparison of **defection-only** vs
**defection + forward view**. Any shift in prey dominance or stability is evidence
of an observation-driven effect (not a change in defection incentives). A clean
conclusion requires the matching defection-only plots.
