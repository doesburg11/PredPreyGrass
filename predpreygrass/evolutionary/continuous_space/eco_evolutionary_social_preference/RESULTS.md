# Social preference: final results and experiment closure

**Closed: 2026-10-28.**

**Conclusion:** a fixed predator attraction/avoidance bias changes behavior.
In the tested settings, the work does not establish an attraction advantage
in survival, reproduction or hunting. Retain the behavioral finding and the
implementation as a baseline; close this phase rather than extend the same
strength/seed searches or add social inheritance without a new hypothesis.

This decision concerns the tested frozen-policy spatial bias. It does not
close the project's broader cooperation or Darwin/Baldwin research, establish
that attraction is universally useless, or demonstrate equivalence between
preference types.

## Scope

Predator preferences −1, 0 and +1 add a directional bias toward or away from
the nearest visible predator to moving-action logits. PPO weights remain
frozen. Preferences are externally assigned, fixed during each lifetime and
non-heritable. This is not an evolved proximity gene, a learned coordination
policy, altruism or reciprocity.

The original ecosystem experiments preserved rewards, physics, body-trait
inheritance, reproduction rules and checkpoint founder pools. The controlled
hunt changed starting populations and poses and used an external stopping
protocol; original capture, grass, energy, reproduction and reward rules
remained intact. Its per-agent action streams were matched across conditions,
rather than replaying the ecosystem's global sampling stream.

## Evidence by phase

| Phase | Design | Finding and limit | Detailed report |
|---|---|---|---|
| Fidelity and directional intervention | Saved networks; neutral action/RNG checks; same-observation counterfactuals | Saved weights and logits restored correctly; directional effect demonstrated. Initial NumPy-sampled pilot trajectories were not seed-equivalent to original inference. | [README: validation and initial results](README.md), [review](REVIEW.md) |
| Realized movement | Checkpoint 1, strength 1, seeds 42–61 | Attraction increased movement toward visible neighbors; overall spacing remained uncertain at this strength. | [Movement](MOVEMENT_RESULTS.md) |
| Strength sweep | Checkpoint 1, strengths 2 and 4, seeds 42–61 | Strength 2 gave closer spacing. A stronger directional bias did not monotonically improve spacing. | [Strength sweep](ATTRACTION_STRENGTH_RESULTS.md) |
| Fresh-seed persistence | Checkpoint 1, neutral versus uniform attraction 2, seeds 62–81 | Closer spacing and longer local encounters; stable nearest-neighbor identity remained inconclusive. | [Persistence](PERSISTENCE_RESULTS.md) |
| Second checkpoint | Checkpoint 2, same uniform comparison, seeds 82–101 | Spacing and encounter-duration effects repeated in a second trained state with its own founder pool. Both checkpoints share an ancestral training run. | [Second checkpoint](SECOND_CHECKPOINT_RESULTS.md) |
| Mixed preferences | Checkpoint 2, lifetime-fixed −1/0/+1, seeds 82–101 | Overall proximity improvement over neutral remained inconclusive; attraction/attraction pairs had longer encounters within the mixed world. This is not proof of partner recognition or fitness. | [Mixed population](MIXED_PREFERENCES_RESULTS.md) |
| Founder outcome pilot | Replay of the 20 mixed episodes | Attraction founders had higher observed life/offspring means, but both intervals included zero. | [Fitness pilot](FITNESS_RESULTS.md) |
| Fresh founder confirmation | Checkpoint 2, 100 mixed episodes, seeds 102–201 | Pilot advantage was not confirmed; both adjusted primary intervals included zero. | [Fitness confirmation](FITNESS_CONFIRMATION_RESULTS.md) |
| Hunting diagnostics | Exact replay of those same 100 mixed episodes | No clear difference in capture rate, retained feeding energy or movement cost. Reused trajectories are not independent replication. | [Hunting](HUNTING_RESULTS.md) |
| Controlled short hunts | Seeds 202–301, three conditions per matched situation, at most 256 ticks | No extra capture successes from attraction over the neutral pair; births ended most trials early. | [Controlled hunts](CONTROLLED_HUNT_RESULTS.md) |

## Behavioral finding retained

For uniform attraction at strength 2:

| Attraction minus neutral | Checkpoint 1, seeds 62–81 | Checkpoint 2, seeds 82–101 |
|---|---:|---:|
| Same-pair close encounter retained for 100 ticks | +4.38 pp [2.42, 6.12] | +5.33 pp [2.93, 7.69] |
| All-predator nearest-neighbor distance | −6.09 units [−8.74, −3.51] | −7.34 units [−10.03, −4.45] |

These are exploratory 95% paired episode-bootstrap intervals. Close means
within 100 world units, with the reports' birth/initial-spell filtering and
censoring definitions. Checkpoint 2's mean nearest-neighbor distance fell
from 109.35 to 102.01 units, approximately 6.7% closer.

The two checkpoints are continuation states from a shared training ancestor,
not independent training-history replicates. Each uses its own founder pool.
Neither these comparisons nor the mixed-population pair associations establish
stable social groups, coordinated hunting, cooperation or social selection.

## Performance advantage not established

The decisive extension beyond the 20-episode founder pilot used 100 fresh
mixed-population seeds, without pooling the pilot. There were two founders
per preference per episode, giving 200 founders per type. Each episode is a
replicate; animals, catches and decisions are not independent replicates.

| Primary attraction minus neutral outcome | Mean difference | Adjusted 97.5% interval |
|---|---:|---:|
| Founder observed time alive | −217.165 ticks | [−785.135, +355.787] |
| Offspring per founder during observed episode | +0.195 | [−0.315, +0.710] |

Both intervals include zero. Ninety-nine episodes reached 8,000 physics ticks;
seed 133 ended at 3,257 with four founders still alive. Those founders remain
right-censored. Observed alive time must not be presented as complete
8,000-tick restricted survival. The predeclared joint confirmation rule failed
because of the intervals and the early-censored founders, not censoring alone.

Read-only hunting measurements on the same trajectories gave:

| Attraction minus neutral, per 1,000 observed predator-ticks | Difference | Exploratory 95% interval |
|---|---:|---:|
| Captures | −0.0090 | [−0.1256, +0.1070] |
| Retained hunting energy | +0.0567 | [−0.2328, +0.3430] |
| Movement energy charged | −0.0536 | [−0.1569, +0.0506] |

Rates average episode-level estimates. These intervals are unadjusted for
multiple comparisons. The results establish neither a hunting advantage nor
the explanation that attraction raises movement costs enough to cancel one.
Body traits, age, prey exposure and survivor composition can affect the rates.

## Controlled hunt limitation

| Outcome, 100 matched situations per condition | One neutral | Two neutral | Two attraction |
|---|---:|---:|---:|
| Captured before competing stop/deadline | 4 | 11 | 9 |
| Trial stopped at first birth | 95 | 89 | 91 |
| Reached 256-tick deadline | 1 | 0 | 0 |
| Mean observed ticks | 38.98 | 36.20 | 36.05 |
| Mean total team predator energy spent | 0.2600 | 0.5415 | 0.5285 |

The attraction-minus-neutral pair comparison was −2 percentage points
(exploratory bootstrap interval −5 to 0). Only two pairs differed in capture
outcome, both favoring neutral; this does not establish harm or equivalence.
The apparent two-versus-one improvement also adds hunting effort and is
uncertain under the report's post-hoc exact paired binary sensitivity check.
It cannot demonstrate cooperation.

Births usually ended trials around tick 37. Consequently this diagnostic
measures capture before birth/other stopping events, not full-length pursuit
success. Cost totals cover unequal observed durations and team sizes.
Success-only capture times condition on different successful subsets. None
should be promoted into a stronger performance claim. A cleaner chase study
would need a separate, explicit resource/reproduction protocol.

## Closure decision

- Close the current fixed-bias proof of concept after documenting the positive
  behavioral effect and the unconfirmed performance benefit.
- Retain code, tests, checkpoints, raw results, plans, figures, source snapshots
  and review records in place. This closure does not delete or archive them.
- Plan no further attraction-strength sweeps, seed extensions, PPO retraining
  or social inheritance within this phase.
- Treat a future coordination or social-evolution study as a new hypothesis
  with a specified ecological mechanism, controls and stopping criteria. Do
  not reinterpret this experiment as evidence for inherited cooperation.
- Continue the broader learning/evolution work separately. The existing
  [field-of-view pilot](../eco_evolutionary_continuous_fov/README.md) addresses
  a different question and is not closed or modified here.

Earlier reports preserve the chronology, including suggestions for next
steps at their dates. This final decision supersedes those suggestions;
completed results and their original provenance remain unchanged.

## Verification and reproducibility

The latest completed verification passed **65 current top-level module tests**:

```bash
.conda/bin/python -m pytest -q predpreygrass/evolutionary/continuous_space/eco_evolutionary_social_preference/test_*.py
```

Explicit top-level selection avoids counting archived source-snapshot tests
as current tests. Checkpoint architecture, every saved tensor and independent
actor logits were validated. Neutral inference and matched replays were
checked; lifetime/offspring bookkeeping and energy conservation were audited.
The 100-episode hunting extension matched every prior saved field and balanced
4,297 individual ledgers. The controlled study verified all 100 matched initial
states and its individual energy ledgers.

Independent Codex CLI reviews, addressed findings and remaining scope limits
are recorded in [REVIEW.md](REVIEW.md) and the phase reports. The controlled
runner's complete source-set guard was hardened after data collection; the
archived run sources and post-run hardening record distinguish those versions.
An audit confirmed that the missing-file edge case did not affect the run.

Each phase report links its exact plans, raw journals, manifests, summaries
and figures. They are the detailed evidence behind this synthesis; no new
simulation or statistical analysis was performed for closure. Historical test
counts in those reports describe their stages, not the final suite size.
