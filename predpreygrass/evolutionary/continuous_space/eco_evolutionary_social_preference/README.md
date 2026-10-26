# Social preference: fixed-policy proof of concept

**Status: implementation prepared; checkpoint evaluation not yet executed.**
Isolated sibling module. The existing continuous-traits code, checkpoints,
training and RESULTS.md are untouched.

## Question

Does a *fixed*, per-agent preference for proximity to visible conspecifics
change the behavior of a frozen predator PPO policy? This is **not** genetic
evolution yet. No social rewards are introduced.

## Mechanism

The original policy produces 48 logits (16 directions × full/half/stop).
For predator actions only, add `strength * preference * cosine(direction,
nearest visible predator)` to moving-action logits; stop-action logits
remain unchanged. Preference -1 = avoidance, 0 = neutral, +1 = attraction.
No visible conspecific => exactly zero bias. Uses the existing predator
observation slots, not hidden environment state.

The direction is global, matching the existing discrete actions. The
preference is applied *after frozen-policy inference and before sampling*.
**Do not use this sampler for PPO training without including the adjusted
log probabilities in PPO's optimization.**

## Run

```bash
python -m pytest predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/test_social_bias.py
python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_social_preference.evaluate /path/to/checkpoint --episodes 5 --output social_poc.json
```

Uses one frozen PPO checkpoint, matched episode seeds, and the same initial
gene pool for -1, 0, +1 preferences. A stochastic run may still diverge
quickly; replicate across independent seeds before drawing conclusions.

## Go / no-go

1. Unit tests: neutrality, no visibility, opposite directional effects.
2. Frozen checkpoint: positive versus negative preference should shift
   mean action alignment in the expected directions.
3. Secondary ecological measurements: normalized distance to visible
   conspecifics and moving-action fraction. These are descriptive, not
   causal proof of improved fitness.
4. If no detectable behavioral effect, stop and debug; **do not add genes**.

No rewards, body genes, selection, policy architecture, or PPO training are
changed in Steps 1–3.
