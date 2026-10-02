# ERL Baldwin with linear SARSA(lambda)

This variant preserves the ecology, genomes, evolution, and ERL/E/L/F/B
comparative conditions from `eco_evolutionary_erl_baldwin`, but replaces its
within-lifetime immediate policy-gradient update with on-policy linear
SARSA(lambda).

The inherited `action_weights` and `action_bias` initialize a newborn's linear
action-value function.  The newborn receives private live copies plus zeroed
eligibility traces.  Learning changes only those live values; reproduction
continues to copy, cross over, and mutate the unchanged genome.  Learned
changes and traces therefore cannot be inherited.

The behavior policy is fixed-temperature softmax over Q-values.  At each
non-terminal transition, the next action is sampled once before the update and
then executed after it, so the bootstrap action and behavior action are
identical as required by SARSA. Death produces one terminal update with zero
bootstrap. Its reward is the final observable evaluator difference plus
`sarsa_terminal_bonus` (default `0.0`), so no ad-hoc death penalty is added.

`sarsa_alpha=0.05` was selected by the development calibration documented in
`RESULTS.md`. The default is SARSA(0): `lambda=0.9` looked competitive at 300
steps but failed the subsequent 3,000-step stability gate on two of three
fresh seeds. The other parameters are not broadly tuned. In particular,
`gamma=1` preserves the endpoint interpretation of the existing
evaluation-difference reward; using `gamma<1` changes that objective by giving
intermediate evaluation levels positive weight.

A subsequent prospective ten-seed confirmation did not validate SARSA(0) as a
replacement for REINFORCE: SARSA(0) went extinct on 4/10 seeds while REINFORCE
survived 10/10. This package remains an experimental implementation and should
not be used for the expensive multi-generation Baldwin run without a new,
independently justified stabilization change and fresh validation.

Run a smoke test with:

```bash
python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.run_sarsa_simulation \
  --steps 1000 --seed 41 --strategy ERL --out-dir /tmp/erl_baldwin_sarsa_smoke
```

Run the matched-seed development calibration with:

```bash
python -m predpreygrass.evolutionary.eco_evolutionary_erl_baldwin_sarsa.calibrate_sarsa \
  --steps 300 --seeds 41,42,43 \
  --out predpreygrass/evolutionary/eco_evolutionary_erl_baldwin_sarsa/calibration.csv
```

This short screen is for rejecting unstable or inert settings and selecting a
candidate for held-out seeds. It is not confirmatory evidence of superiority.
