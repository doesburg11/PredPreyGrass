# eco_evolutionary_continuous_traits: results

## Runs

| Run | Folder (`runs/eco_evolutionary_continuous_traits_...`) | Traits | Iterations | Simulation workers | Wall time | Status |
|---|---|---|---:|---:|---:|---|
| CT-01 | `2026-10-15_21-31-22_924979` | expressed | 3,000 | 14 | 21.3 h | completed |
| CT-02 | `2026-10-15_21-31-42_894279` | neutral control | 3,000 | 14 | 13.7 h | completed |

Both: every quirk fixed, cost coefficients speed 1.0, vision 0.5, field of
view 0.25 (SPEC.md section 11), trained side by side from scratch. A first
attempt at the pair (2026-10-15, 20:00) was stopped after about 200
iterations to speed up the contact search (commit af36d91); its logs are
kept as `runs/train_ct0*_2026-10-15_20-*.aborted.log`.

## 1. Pilot: traits expressed against the neutral control (CT-01, CT-02)

**Question.** Do the traits evolve by selection rather than drift, and does
an arms race in bodies appear next to the learning arms race?

**Measures.** Mean gene of the living animals at the end of every episode,
and the selection differential `cov(offspring / mean offspring, gene)` over
every animal of the episode (SPEC.md section 11), both averaged by RLlib
over recent episodes and logged per iteration; a 50-episode final check
and a tournament of checkpoints 300–2,990 (every 300, 100 episodes per
matchup, seeds 10,000–10,099), each species playing with the gene pool of
its own checkpoint. [`analysis/pilot.py`](analysis/pilot.py) writes
[`analysis/results/pilot.json`](analysis/results/pilot.json) and the figure.

![Pilot: mean genes and selection differentials over training, CT-01 against CT-02](../../../assets/images/readme/continuous_traits_pilot.png)

**Genes and selection** (mean gene at the end of training; selection
differential averaged over all iterations, and the share of iterations in
which it is positive):

| | CT-01 mean gene | CT-01 selection (share > 0) | CT-02 mean gene | CT-02 selection (share > 0) |
|---|---:|---:|---:|---:|
| Predator speed | −0.50 | −0.0031 (0%) | +0.12 | −0.0001 (59%) |
| Predator vision | −0.61 | −0.0067 (0%) | −0.03 | +0.0003 (81%) |
| Predator field of view | +0.12 | +0.0007 (95%) | −0.01 | −0.0003 (38%) |
| Prey speed | −0.10 | +0.0004 (100%) | 0.00 | +0.0001 (61%) |
| Prey vision | −0.47 | −0.0036 (0%) | −0.04 | −0.0001 (35%) |
| Prey field of view | +0.50 | +0.0031 (100%) | −0.02 | +0.0001 (66%) |

Genes are log-ratios to the species' base value, limited to ±0.7: CT-01's
predators end at about 0.61 × the base top speed and 0.54 × the base view
distance, its prey at 0.63 × the base view distance and 1.65 × the base
field of view.

**Final checks** (50 episodes each; BE-04 and BE-05 of
`base_environment_continuous`, the same world without traits, for
comparison):

| Run | Coexisting | Prey extinct | Predators extinct | Catch risk | Predators / prey alive on average | Predators stopped | Prey stopped | Catches by a standing predator |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| CT-01 | 13 | 36 | 1 | 1.18 | 19.5 / 44.4 | 65% | 67% | 10% |
| CT-02 | 46 | 1 | 3 | 1.53 | 10.0 / 32.0 | 46% | 7% | 2.5% |
| BE-04 | 47 | 1 | 2 | 1.93 | 10.6 / 27.1 | 44% | 7% | 2% |
| BE-05 | 46 | 1 | 3 | 1.69 | 10.4 / 28.8 | 48% | 3% | 2% |

**Tournaments** (share of comparisons against a fixed opponent in which
the newer checkpoint does better, and mean catch-risk change late in
training; 95% paired-seed bootstrap intervals):

| Run | Prey, all | Predators, all | Prey risk reduction, late | Predator risk increase, late | Episodes ending in prey extinction |
|---|---:|---:|---:|---:|---:|
| CT-01 | 41% (39–48) | 40% (38–44) | 0.01 (−0.01–0.03) | 0.00 (−0.03–0.02) | 70% |
| CT-02 | 93% (88–93) | 82% (78–84) | 0.14 (0.06–0.23) | 0.05 (−0.02–0.12) | – |

In CT-01, catch risk is about 1.0–1.5 for every pairing of checkpoints,
with no trend along either axis.

**Results.**

- **The traits evolved by selection in CT-01.** Predator speed and vision
  and prey vision fell, prey field of view rose, each with a selection
  differential of one sign in every iteration (0% or 100% positive) and
  5–30 times larger than in CT-02. In CT-02, where the genes have no
  effect, the selection differentials change sign and the means wander
  (predator speed reached +0.25 and came back to +0.12): drift, which in one
  run can move a gene mean by a third of its range.
- **Cheap bodies, not better ones.** CT-01's predators became slow and
  short-sighted, cutting their resting cost to 0.44 of the base; its prey
  traded view distance for a wider field of view. Predator vision stayed
  near −0.62 for 2,000 iterations while its selection differential stayed
  negative, so the gene limit (−0.7, where mutation can only push up) is
  probably what holds it there.
- **A different, less stable ecology.** At episode ends, CT-01 had 34
  predators and 3 prey on average over its last 500 iterations, against 12
  and 22 in CT-02. In the final check both species survived in 13 of 50
  episodes (CT-02: 46), and prey died out in 36. Both species stood still
  about two thirds of the time (prey in CT-02: 7%).
- **No learning arms race in CT-01.** Newer checkpoints did not do better
  against fixed opponents (41% and 40%, both intervals below 50%), unlike
  CT-02 and all five runs of `base_environment_continuous`.

**Interpretation and caveats.** One run per condition. Within that run, the
evidence points to selection on the traits, with a much larger selection
differential than drift produced in the control. A plausible reading, not
tested: with the chosen costs, capacity was so expensive that the cheapest
bodies won; cheap predators became numerous enough to keep prey near
extinction, and with outcomes set by numbers rather than skill, neither
species gained from improving its behaviour. That reading depends on cost
coefficients chosen as starting guesses. Two further confounds: observations
scale distances by the observer's own view distance and speeds by the
observed animal's own top speed, so the shared policy may misread animals
with unusual bodies, which could itself shape selection on vision and
speed; and by the first tournament checkpoint (iteration 300) the bodies had
already partly evolved, so the tournament does not show the transition.

**Next.** Make observations use fixed scales (distances / 400, speeds / 12)
and run cheaper capacity: a new neutral control and cost coefficients at
0.5× and 0.25×. A repeat at 1× would show whether the collapse of the arms
race and of coexistence repeats, which would be of interest in itself
(predators evolving to over-exploit their prey).
