# Stag Hunt vs Defection: Evaluation Comparison (1000-step runs)

This note compares the non-defection `stag_hunt` environment with the
`stag_hunt_defection` variant using the 1000-step evaluation plots you provided.
It is a qualitative, plot-based analysis (not a full statistical study).

## Inputs used

- Base env: `src/predpreygrass/rllib/stag_hunt/predpreygrass_rllib_env.py`
- Base config: `src/predpreygrass/rllib/stag_hunt/config/config_env_stag_hunt.py`
- Base eval plots (1000 steps):
  - `src/predpreygrass/rllib/stag_hunt/ray_results/STAG_HUNT_PRED_LOSS_0_08/PPO_PredPreyGrass_3c565_00000_0_2026-01-01_18-06-38/checkpoint_000019/eval_multiple_runs_PRED_DECAY_0_20_PRED_OBS_RANGE_9_GRID_30_INITS_15_2026-01-03_19-46-32/summary_plots/iters=1000`

- Defection env: `src/predpreygrass/rllib/stag_hunt_defection/predpreygrass_rllib_env.py`
- Defection config: `src/predpreygrass/rllib/stag_hunt_defection/config/config_env_stag_hunt_defection.py`
- Defection eval plots (1000 steps):
  - `src/predpreygrass/rllib/stag_hunt_defection/ray_results/STAG_HUNT_DEFECT_PRED_LOSS_0_08_2026-01-05_01-26-09/PPO_PredPreyGrass_225cc_00000_0_2026-01-05_01-26-10/checkpoint_000029/eval_multiple_runs_STAG_HUNT_DEFECT_2026-01-05_14-04-28/summary_plots/iters=1000`

Legend reference (per `stag_hunt/README.md`):
- `type_1_prey` = mammoths
- `type_2_prey` = rabbits

## Qualitative findings from the plots

### Base `stag_hunt` (no defection)

- In 3 of 4 plots, mammoths (`type_1_prey`) crash early and stay near zero.
- Rabbits (`type_2_prey`) persist and stabilize at moderate levels.
- One run shows the opposite (mammoths persist while rabbits collapse), so the
  pattern is dominant but not absolute.

### `stag_hunt_defection`

- In all 4 plots, rabbits crash early and stay low.
- Mammoths persist at moderate levels through most of the run.
- Predator counts look slightly higher and more stable in mid/late steps.

## Interpretation

### 1) Stag hunt principle

In the base environment, capture is effectively always cooperative because all
nearby predators contribute. That makes mammoth hunting relatively reliable and
can drive mammoths to extinction early. This shifts the ecology toward rabbits
(the lower-energy prey) as the stable option.

With defection enabled, only joiners count toward capture and they pay a cost.
That reintroduces coordination risk and reduces effective mammoth kill rates,
allowing mammoths to persist. The dynamics look closer to the stag-hunt dilemma:
high-payoff prey can survive when cooperation is less reliable.

### 2) Cooperation vs defection

Defection does not eliminate cooperation (join rates were still around 80% in
the sample metrics), but it makes cooperation less effective at consistently
securing high-energy prey. Predators can rely more on rabbits for immediate
energy, which likely overexploits rabbits and pushes them to extinction while
mammoths survive.

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

The plots support the claim that adding defection shifts the ecology from a
rabbit-dominant regime toward mammoth persistence. In other words, voluntary
participation reintroduces the coordination fragility expected in a stag-hunt
setting, which makes high-payoff prey harder to eliminate and changes the
long-run prey mix.
