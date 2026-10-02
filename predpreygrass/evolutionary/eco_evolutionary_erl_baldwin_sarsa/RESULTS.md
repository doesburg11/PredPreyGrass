# Calibration results

## 2026-10-02: matched-seed 300-step population screen

This is a short algorithm calibration, not a Baldwin-effect or long-term
survival result. Every treatment used the unchanged full-scale ecology and 60
founders. The primary screening outcome was mean living-agent count across
steps 0-300; population at step 300, births, end-of-run survivor parameter
displacement, final survivor TD magnitude, Q magnitude, and trace magnitude
were also recorded. No run was
discarded. Raw rows are in `calibration.csv`, `calibration_heldout.csv`, and
`calibration_heldout_extra.csv`.

Development seeds 41-43 compared REINFORCE, evolution-only, and the complete
grid `alpha in {0.001, 0.01, 0.05}` by `lambda in {0, 0.5, 0.9}`. The selected
setting was `alpha=0.05, lambda=0.9`: it had the highest development mean
population trajectory (235.2), versus 133.5 for REINFORCE and 191.1 for
SARSA(0) at the same alpha. All settings remained finite.

The learning rate was then frozen at 0.05. Seeds 44-46 were run first, and
their wide interval was inspected before the unchanged comparison was extended
to seeds 47-51. Thus all seeds 44-51 are held out from hyperparameter tuning,
but the pooled eight-seed inference is sequential/exploratory rather than a
strictly preregistered confirmatory test. These seeds compared only REINFORCE,
evolution-only, SARSA(0), and SARSA(0.9). Paired differences in mean population
trajectory were:

| Contrast | Mean difference | Paired 95% t interval |
|---|---:|---:|
| SARSA(0.9) - REINFORCE | +68.7 | +38.0 to +99.3 |
| SARSA(0) - REINFORCE | +65.8 | +14.3 to +117.4 |
| SARSA(0.9) - SARSA(0) | +2.8 | -43.2 to +48.9 |

There were no extinctions or numerical failures. The held-out result supports
linear SARSA with `alpha=0.05` as a promising alternative to the current
immediate policy-gradient learner over this short horizon. It does **not** show
that eligibility traces add value: no trace-specific difference was detected
in these eight seeds, and the interval remains compatible with effects from
about -43 to +49 agents. The variants also changed rank across seeds. This
does not establish a
Baldwin effect, long-term stability, or genetic assimilation. Population count
can reward rapid reproduction followed by a later crash, so the selected
settings must next survive longer runs before any multi-generation conclusion.

The calibrated default is therefore `alpha=0.05`, while `lambda=0.9` remains a
predeclared candidate rather than a demonstrated improvement. Future runs
should retain SARSA(0) as a required control.

## 2026-10-02: 3,000-step stability gate

Fresh matched seeds 52-54 were run without retuning. The four frozen
treatments were REINFORCE, evolution-only, SARSA(0), and SARSA(0.9), with both
SARSA variants using `alpha=0.05`. Runs ended at step 3,000 or extinction.
Mean-population calculations treat extinction as an absorbing zero through the
common 3,000-step horizon. Raw results are in `stability_3000.csv`.

| Treatment | Extinctions | End population, mean | Trajectory population, mean | Mean peak | Mean carnivore peak |
|---|---:|---:|---:|---:|---:|
| REINFORCE | 0/3 | 93.3 | 162.0 | 549.7 | 118.3 |
| Evolution-only | 2/3 | 256.0 | 105.2 | 379.7 | 50.3 |
| SARSA(0) | 0/3 | 338.7 | 186.5 | 651.3 | 144.7 |
| SARSA(0.9) | 2/3 | 243.0 | 136.5 | 640.3 | 209.0 |

SARSA(0.9) showed the feared boom-and-bust pattern. It peaked at 523 and 498
agents on seeds 52 and 53, drove carnivore peaks of 253 and 146, then went
extinct at steps 684 and 1,030. Seed 54 survived with 729 agents, so the trace
variant is strongly seed-fragile rather than uniformly broken. Its maximum
observed absolute Q and trace values remained finite (3.13 and 9.79); the
failure was ecological, not a numerical divergence.

SARSA(0) survived all three seeds through step 3,000, as did REINFORCE. Its
maximum observed absolute Q value stayed below 1.83 and it never reached the
2,000-agent safety cap. It had the highest end population and a modestly higher
mean population trajectory than REINFORCE, but three seeds are insufficient to
claim superiority. The provisional engineering decision for this configuration
is that eligibility traces (`lambda=0.9`) failed this three-seed gate, while
one-step linear SARSA (`lambda=0`) passed it. This is not a general statistical
result about eligibility traces.

`carnivore_mean` in the raw CSV is calculated only over the observed portion of
a run before stopping, so it is not comparable across treatments when extinction
occurs. The table uses carnivore peak, not that observed-horizon mean.

The default is therefore changed from `lambda=0.9` to `lambda=0.0`. The next
comparison should use more fresh seeds and survival-focused endpoints for
SARSA(0) versus REINFORCE; SARSA(0.9) should remain only as the documented
failed-trace control, not proceed to an expensive Baldwin study.

## Prospective confirmation protocol

Before inspecting the next results, the confirmation was fixed as ten fresh,
matched seeds (55-64), 3,000 steps per run, `alpha=0.05`, and `lambda=0`. The
primary safety endpoint is the number of extinctions. The secondary endpoint is
adaptive-agent population averaged over the common 3,000-step horizon, with
post-extinction values treated as zero. End population, peak population,
carnivore peak, and numerical diagnostics are descriptive only. The decision
comparison is SARSA(0) against REINFORCE; evolution-only is retained because the
calibration runner emits it as a contextual control.

## Prospective confirmation result

Raw results are in `confirmation_3000.csv`; the runner-generated aggregate is
in `confirmation_3000.summary.json`. There were no numerical failures and no
population-cap events.

| Treatment | Extinctions | End population, mean | Trajectory population, mean | Mean peak | Mean carnivore peak |
|---|---:|---:|---:|---:|---:|
| REINFORCE | 0/10 | 309.4 | 222.4 | 717.8 | 128.3 |
| Evolution-only | 8/10 | 36.3 | 59.3 | 235.0 | 35.9 |
| SARSA(0) | 4/10 | 206.6 | 205.4 | 804.9 | 202.8 |

SARSA(0) became extinct on seeds 60, 61, 62, and 64, at steps 710, 994,
1,358, and 841 respectively. REINFORCE survived every matched run. With only
four discordant pairs, a two-sided exact McNemar test is necessarily weak
(`p=0.125`), but the observed direction fails the predeclared safety gate and
does not support advancing SARSA to the expensive multi-generation study.

For the secondary endpoint, the paired SARSA-minus-REINFORCE difference in
common-horizon mean population was -17.0 agents (95% t interval -108.6 to
74.7; paired t-test `p=0.685`). The result provides no detected population
advantage that could offset the extinction signal. SARSA's higher average peak
and carnivore peak are consistent with a more aggressive boom-and-bust ecology.

The three-seed stability result was therefore a false reassurance. The current
SARSA formulation is retained as a tested negative result, not recommended as
the replacement learner. A multi-generation run should not be started with it
as currently configured.
