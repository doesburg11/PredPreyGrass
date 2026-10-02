# Training Analysis — eco_evolutionary_erl_baldwin_hebbian

This module was forked from `eco_evolutionary_erl_baldwin`
on 2026-10-01 to test whether its confirmed result (ERL significantly beats
evolution alone / learning alone / no adaptation, p<0.00001 at n=100/condition)
depends on that module's specific learning algorithm (a REINFORCE-style
policy-gradient update) or generalizes to a structurally different one
(reward-modulated Hebbian plasticity) — see README.md's "Why this module
exists" for the full framing.

Current state: the mechanism is implemented, unit-tested, and
smoke-tested (all 11 strategies run without crashing; a 1500-step run at the
paper's default 100x100 config showed `hebb_trace_absmean` moving measurably
away from zero under "ERL," confirming the plasticity channel is actually
active, not dead code). **No comparative study has been launched.**
`hebbian_eta`, `hebbian_trace_clip`, and the alpha-coefficient init scale are
all untuned first guesses (see config.py) — before trusting any trained
result, check whether these need retuning the way ERL Baldwin's own
energy/reproduction constants did (see that module's RESULTS.md §1-§8 for the
retuning history this world's mechanics already went through, which this fork
inherits unchanged).

For the ORIGINAL module's full results (the ERL result itself, and the
C/ERLC, K/ERLK, S/ERLS dead-end diagnoses this fork's README explicitly flags
as inherited-but-not-re-verified under Hebbian plasticity), see
`../eco_evolutionary_erl_baldwin/RESULTS.md`.

## Uniform-positive-alpha control

The random-sign scrambling hypothesis was tested with a first-class
`action_alpha_mode="uniform_positive"` control. For each genome, its behavioral
weight-alpha and bias-alpha tensors are replaced by positive constants equal
to their respective genomic mean absolute magnitudes. The genome itself is
unchanged. This removes signs and per-site heterogeneity while holding the
average coefficient magnitude fixed.

Ten matched standardized lifetime probes (seeds 41-50, 3,000 decisions each)
gave both modes the same sequence of cardinal threat observations and the same
pre-generated action-sampling uniforms. A fixed evaluator rewarded reductions
in the threat signal: directly away cleared it, perpendicular halved it, and
toward left it unchanged. This counterfactual assay avoids divergent ecology
trajectories and treatment-dependent encounter counts. Raw results are in
`uniform_alpha_control.csv`.

| Routing | Mean away-action rate | Mean effective displacement |
|---|---:|---:|
| Genomic random-sign | 23.73% | 0.139 |
| Uniform positive | 23.93% | 0.124 |

Uniform-positive routing improved the away-action rate in 7/10 matched seeds,
but the mean paired improvement was only 0.21 percentage points (95% t interval
-0.07 to 0.48; paired t-test `p=0.121`, Wilcoxon `p=0.160`). It did not produce
a larger final effective-weight displacement on average.

Conclusion: removing the random signs may help somewhat, but it did not
reliably unlock the mechanism and several policies still never moved their
argmax consistently. The standardized single-lifetime gate did not pass, so
the planned population-level control was not launched. Random-sign scrambling
is not supported as the sole or sufficient explanation for the Hebbian
learner's failure.
