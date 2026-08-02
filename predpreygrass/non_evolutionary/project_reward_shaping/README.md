# Reward shaping: sparse vs. dense rewards

This folder holds one connected line of investigation: starting from a
single question ("is the base environment's sparse reward hurting
training?"), it grew into five trained sibling environments and a headline
result that reverses the question it started with — **reward shaping should
be minimized here, not maximized.** This README is the full story:
motivation, methodology, every module's result, the mechanistic
explanation, and what's still open.

Each module below has its own README with implementation-level detail
(exact reward mechanics, config). This file is the overview and the
results log — read this first.

## 1. Motivating hypothesis

[`base_environment`](../base_environment)'s only nonzero reward anywhere is
a flat `+10` bonus at the instant of successful reproduction — every other
reward hook (`reward_predator_catch_prey`, `reward_prey_eat_grass`,
`reward_predator_step`, `reward_prey_step`, `penalty_prey_caught`) is `0.0`.
That means every agent gets zero training signal for the ~50-200+ steps
between reproduction events.

This investigation started as the chosen next step after the Darwin/Baldwin
evolutionary project's Trial 7 came back null: rather than continuing to
retune/reseed the same shape of experiment, the plan was to first fix
reward sparsity as groundwork for a future, more progressive attempt at
combining Nature (heritable genome) and Nurture (lifetime RL learning). The
hypothesis: this sparsity was hurting training — slow, chaotic early
learning — documented across every module's `RESULTS.md` repo-wide.
Replacing it with a **dense, biologically literal reward** (each agent's
reward equal to its own net energy delta every single step — decay,
movement, eating, reproduction cost, no hand-designed shaping constants)
was expected to improve outcomes.

**This hypothesis is what the final result below falsifies.**

## 2. Methodology

Every module below was trained a full **1000 PPO iterations** under
identical hyperparameters and resource configuration, so any difference in
outcome is attributable to reward design, not infrastructure:

- `gamma=0.99`, `lr=0.0003`, `train_batch_size_per_learner=1024`,
  `minibatch_size=128`, `num_epochs=30`, `entropy_coeff=0.0`,
  `clip_param=0.3`, `kl_coeff=0.2`, `kl_target=0.01`
- `num_gpus_per_learner=1`, `num_learners=1`, `num_env_runners=20`
- Same conv/FC architecture for both predator and prey policies
  (`[16,32,64]` conv filters, `[256,256]` FC)

**Sequential, not concurrent, training.** Two runs were briefly trained
concurrently to save wall-clock time; this pushed combined GPU memory to
~92% early, before this environment's known memory-growth-with-episode-
length pattern even kicked in — a real OOM risk, and Ray does not
coordinate GPU memory across independent clusters (it's shared via normal
OS/CUDA time-slicing like any two unrelated processes). Since every run
uses identical batch size and hyperparameters, "iteration N of run A" vs.
"iteration N of run B" is a valid comparison regardless of wall-clock
simultaneity — concurrency was pure convenience, not a validity
requirement. Every run after the first two was trained solo,
full-resource, sequentially.

**Reproduction counts, not raw reward, as the comparison metric.** Raw
`episode_return_mean` is not comparable across these modules — the reward
*scales* are fundamentally different (discrete `+10` spikes vs. continuous
per-step deltas of very different magnitude). Every comparison below uses
real, reward-scheme-independent behavioral outcomes instead: births per
species (predator/prey reproduction counts) and final population size,
measured by running the final trained checkpoint through several seeded
episodes with deterministic actions.

## 3. Results table

| module | reward design | predator births (avg, 3 seeds) | prey births (avg) | % of sparse (pred / prey) | wall time |
|---|---|---|---|---|---|
| [`base_environment_sparse_rewards`](base_environment_sparse_rewards) | sparse, `+10` on reproduction only | **135.3** | **588.7** | 100% / 100% | **12.45h** |
| [`base_environment_sparse_rewards_plus_eating`](base_environment_sparse_rewards_plus_eating) | sparse + asymmetric eating bonus (`+1` predator / `+0.1` prey) | 111.3 | 552.0 | 82% / 94% | 11h22min |
| [`base_environment_dense_rewards_additive`](base_environment_dense_rewards_additive) | dense per-step energy delta **+** `+10` reproduction bonus | 85.0 | 445.0 | 63% / 76% | 16.22h |
| [`base_environment_dense_rewards`](base_environment_dense_rewards) | dense per-step energy delta only, no reproduction bonus | 56.7 | 311.0 | 42% / 53% | 18.68h |
| [`base_environment_sparse_rewards_plus_kickback`](base_environment_sparse_rewards_plus_kickback) | sparse + `+10` grandparent kick-back on grandchild birth | 117.0 | 562.3 | 86% / 96% | 14.44h |

Sparse wins on every axis measured against every other variant: highest
reproduction rate for both species, most balanced final predator:prey ratio,
zero extinction events across all tested seeds (pure dense had one predator
population go fully extinct even at its final, fully-trained checkpoint),
and the fastest wall-clock time despite supporting the largest population
(an anomaly not fully explained — see section 5). Kickback is the closest
runner-up of the four shaping variants — the only one to beat sparse+eating
on *both* axes — but still falls short of sparse itself on both.

## 4. Module by module

### `base_environment_sparse_rewards` — the baseline

Byte-for-byte the same sparse, reproduction-only reward as
`base_environment`. Exists to be the fair comparison partner for every
other module here. **Result**: 135.3 / 588.7 births, zero extinctions
across tested seeds, first reached ~1000-step episodes at iteration 20.

### `base_environment_dense_rewards` — pure dense replacement

Reward is pure per-step net energy delta:
`reward = energy_after - energy_before` (folds in decay, movement,
eating, reproduction cost). No reproduction bonus at all — the direct test
of the original hypothesis. **Result**: worst of all five — 56.7 / 311.0
births (42%/53% of sparse), one predator-extinction event observed in 3
tested seeds even at the final, fully-trained checkpoint. Also the slowest
wall-clock despite the smallest population.

### `base_environment_dense_rewards_additive` — dense + reproduction bonus

The dense-pure result raised an obvious question: is pure dense losing
because of *density*, or simply because it drops the reproduction
incentive entirely (reproduction here isn't an action an agent takes —
it fires automatically on crossing an energy threshold — so a pure energy
signal doesn't distinguish "accumulate energy" from "accumulate energy in
order to reproduce")? This module layers the sparse variant's `+10`
reproduction bonus on top of the dense per-step delta (additive, not
replacement). **Result**: recovers most but not all of the gap — 85.0 /
445.0 births (63%/76% of sparse), still short of the pure sparse baseline
and still slower wall-clock than sparse despite fewer agents.

### `base_environment_sparse_rewards_plus_eating` — isolating the noise vs. incentive question

The additive result sharpened the question further: was reward *density*
itself the problem, or specifically the continuous per-step signal's noise
sitting in the *same reward channel* as the reproduction event? (Concretely
demonstrated: the additive variant's reproduction-step rewards were
observed scattered across ~9.3–12.7 rather than sparse's exact, invariant
`10.0`, because the dense delta rides along underneath the flat bonus.)

This module tests that directly: the same clean, event-based sparse-reward
style as the baseline (zero continuous signal, zero decay/movement terms in
the reward at all), with one more *discrete* event type rewarded — eating —
alongside reproduction. The eating reward is deliberately **asymmetric**,
`+1` predator / `+0.1` prey, not a flat `+1`/`+1`: measured directly by
running the sparse baseline's final trained checkpoint (3 seeds, full
1000-step episodes, counting real eating events), predators catch prey
~4.4 times per reproduction on average, but prey eat grass ~60.5 times per
reproduction (grass regrows slowly — `energy_gain_per_step_grass=0.04`,
capped at `initial_energy_grass=2.0` — and gives little per visit, so prey
need many more, smaller meals to reach their reproduction threshold). A
flat `+1` for both would make prey's *total* eating reward per reproduction
cycle (`60.5 × 1 = 60.5`) six times larger than the reproduction reward
itself (`10.0`), swamping the primary incentive the same way the dense
signal's noise did. `+1`/`+0.1` keeps each species' total eating reward per
cycle clearly secondary to reproduction for both (predator: `4.4×1=4.4`;
prey: `60.5×0.1≈6.05`).

**Result**: recovered **82% (predator) / 94% (prey) of sparse's
reproduction rate** — a much better recovery than dense-additive ever
achieved, despite both adding a comparably-sized secondary incentive on top
of the same reproduction bonus. Stable from early in training (checkpoint
~290/1000 already showed 83%/93%) through the final checkpoint — not a late
fluke. Strong support for "clean discrete signals cost little, continuous
ones cost a lot" specifically, not just "less shaping is better" in
general.

### `base_environment_sparse_rewards_plus_kickback` — grandparent kick-back

A further design discussion (after the eating-bonus result) landed on
testing kin-selection-style reward: keep the `+10` reproduction reward
unchanged, and add a second `+10` **kick-back** to a grandparent every time
its own child successfully reproduces (i.e. every time a grandchild is
born). Fires repeatably — once per grandchild, not capped at one per
lineage — and only if the grandparent is still alive to collect it (RLlib
cannot deliver a new reward to an agent already marked `terminated=True`).

This mechanism (`_reward_parent_for_child_reproduction`) had already been
tried elsewhere in this repo's history, in a more complex two-predator-
type/two-prey-type environment with walls and occlusion.
That prior attempt was tested at `kin_kick_back_reward = 4.0` (~0.4× the
`10.0` reproduction reward) and found no benefit. This module reimplements
the same mechanism in the single-predator/single-prey-type
`base_environment_*` family instead (directly comparable to the other four
runs here) and tests it at a full **1:1 weight** (`kickback_reward = 10.0`)
— a genuinely different, untested magnitude. Whether the earlier null
result was due to the weaker magnitude or the richer environment is an
open question this module's result doesn't resolve on its own, since both
changed at once.

**Result**: trained the full 1000 iterations (14.44h, completed
2026-08-01), then evaluated the final checkpoint the same way as every
other module here — 3 seeded, full 1000-step episodes, deterministic
actions, births counted via each species' monotonic newborn-ID counter.
**117.0 predator / 562.3 prey births (86%/96% of sparse)**, zero
extinction events across all 3 seeds, every episode ran the full 1000
steps without early collapse.

This makes kickback the best-recovering of the four shaping variants on
*both* axes — ahead of `sparse_rewards_plus_eating` (82%/94%) despite
adding a secondary reward at a much larger magnitude (`+10`, full 1:1
weight vs. eating's asymmetric `+1`/`+0.1`). That's a genuine wrinkle in
the section 5 "clean discrete signal costs little" explanation: magnitude
alone doesn't predict the damage a secondary signal does. See section 5's
"Why kickback recovers more than eating" for the mechanistic explanation —
in short, it isn't that eating and reproducing are opposed goals (eating is
strictly necessary for reproduction here), it's that eating fires far more
often than reproduction, so its reward contributes a large, non-negligible
fraction of total return per reproductive cycle in a way that isn't
guaranteed to leave the optimal policy unchanged (Ng, Harada & Russell,
1999). Kickback's payout, by contrast, is gated by the same rare event
class as the primary reward (reproduction, one hop removed via kinship),
so it can't inflate return the same way. This reading is plausible, not
confirmed — n=1 training run, same caveat as every other result in this
table (see section 7).

**Still short of pure sparse on both axes, though** (86%/96%, not
100%/100%) — being "the least damaging of the shaping variants" is not the
same as "reward-neutral," and the reason is a second, distinct mechanism:
see section 5's "Why kickback still falls short of sparse" for the
credit-assignment argument (kickback pays the *grandparent* for the
*child's* reproduction — the receiving agent didn't cause the triggering
event). This doesn't overturn the headline finding — see section 8.

## 5. Cross-cutting findings

**Episode-length ramp-up is identical across every variant.** How quickly
agents learn to survive a full 1000-step episode at all was a dead heat —
every module tested so far first reached that point at the *identical*
training iteration: 20. Reward density and reward design have shown zero
measurable effect on this axis in this environment.

**Wall-clock ordering is not fully explained.** Sparse is the fastest of
all runs despite supporting the largest final population (more agents
should mean more per-step compute). This is a real, measured pattern, not
a mechanistically confirmed one — flagged as an open question, not a
settled explanation.

**Why sparse wins (best current explanation).** Classic "sparse reward is
hard" problems in RL are usually about *delayed* credit assignment —
reward arriving long after the actions that caused it. The reproduction
reward here isn't delayed; it fires immediately on the step reproduction
happens, it's just infrequent. PPO's value function is specifically built
to bootstrap across gaps like that (`gamma=0.99` gives an effective horizon
of ~100 steps, in range of the ~50-200 step gap the original hypothesis was
worried about). What the hypothesis didn't anticipate: layering a
continuous per-step signal into the *same reward channel* as the
reproduction event makes that one important signal noisier and harder for
PPO to cleanly attribute — even when the reproduction bonus is explicitly
restored on top (additive recovers 63-76%, doesn't fully close the gap).
The eating-bonus result reinforces this: a second *discrete* signal in a
separate, clean event channel costs comparatively little (82-94%
retention), even at a similar total secondary-incentive magnitude to
additive. In short: the sparsity itself wasn't the problem; adding density
introduced a different cost — signal noise — that outweighed the benefit it
was meant to provide.

**A biological-realism framing, not just an RL-engineering one.** Minimizing
hand-designed reward shaping also happens to align with modeling literal
Darwinian fitness: reproductive success, not a proxy signal for it. The
cleanest-performing designs so far (sparse baseline, sparse+eating,
kickback) are also the ones closest to "reward = did you (or your lineage)
reproduce" rather than "reward = a continuous approximation of how well
things are going."

**Why kickback recovers more than eating, despite a larger magnitude — reward-shaping theory, not "competing incentives."**
Kickback's secondary reward is the largest tested (`+10`, equal to the
primary reproduction bonus) yet it recovered *more* of sparse's
reproduction rate than eating's much smaller, asymmetric bonus did (86%/96%
vs. 82%/94%). It's tempting to explain this as eating and reproducing
being "competing" goals — but that framing doesn't hold up: eating is
strictly *necessary* for reproduction in this environment (reproduction
fires automatically once energy crosses a threshold; there is no way to
reach that threshold without eating). Rewarding eating rewards a real,
required step toward the true goal, not a distraction from it. So why does
it still cost something sparse and kickback don't pay?

The relevant theory is Ng, Harada & Russell's (1999) result on **policy
invariance under reward transformations** ("Policy invariance under reward
transformations: Theory and application to reward shaping," ICML 1999).
Their theorem: adding a shaping term `F(s, a, s')` to an environment's
reward is *guaranteed* not to change which policy is optimal only if `F`
is **potential-based** — expressible as `F(s, a, s') = γΦ(s') − Φ(s)` for
some potential function `Φ` over states. A shaping term of that form
telescopes over any trajectory: its sum from step 0 to step `T` collapses
to `γ^T Φ(s_T) − Φ(s_0)`, a boundary term that depends only on where the
trajectory started and ended, not on the path taken between — so it can
change *how fast* an agent learns without ever changing *what* the optimal
policy is. A flat "+1 every time event X happens" bonus is generically
**not** of this form: it doesn't telescope, it just accumulates once per
occurrence of X, and the theorem's guarantee simply does not apply to it.
That doesn't mean it *must* distort the optimal policy — only that nothing
protects it from doing so.

Whether it actually does, and how much, comes down to how large a share of
total return the shaping term contributes, and how tightly its firing rate
tracks the true objective (reproduction) versus some other quantity. This
environment's own numbers (section 4, `sparse_rewards_plus_eating`'s
writeup) make the comparison concrete:

| | fires per reproduction cycle (avg) | reward per firing | contribution per cycle | as % of total cycle reward |
|---|---|---|---|---|
| predator eating | 4.4 catches | `+1` | `+4.4` | 4.4 / 14.4 ≈ 31% |
| prey eating | 60.5 grazes | `+0.1` | `+6.05` | 6.05 / 16.05 ≈ 38% |
| kickback | ~1 grandchild birth per own reproduction (structurally) | `+10` | `+10` | bounded by the same reproduction-event rate, not a separate faster-firing behavior |

For prey specifically, **eating contributes roughly a third of the total
reward in a typical reproductive cycle** — not a rare top-up riding along
next to the `+10`, but a substantial, frequently-arriving reward stream in
its own right. Because that stream fires on *every* catch/graze regardless
of whether that particular meal was on the fastest path to the next
reproduction event, a policy can pick up extra return by eating somewhat
more than strictly necessary — additional opportunistic catches, less
time-efficient foraging, more caution/less urgency once "comfortably fed"
— without that behavior needing to *also* maximize the rate of reaching the
reproduction threshold. Nothing in the shaping term penalizes that
divergence; potential-based shaping would (by construction, via the
telescoping property), an unconstrained per-event bonus does not.

Kickback's payout structurally cannot be inflated the same way: it never
fires on an intermediate, frequently-repeatable behavior at all — only on
an instance of the *same rare event class* the primary reward already
targets (a reproduction event, one hop removed via kinship). There is no
"do a bit more of the rewarded thing than optimal" failure mode available
when the only rewarded thing is more reproduction, however indirectly.
That is the actual mechanistic distinction — a frequency/base-rate
argument grounded in a specific, citable non-invariance result, not a
vaguer "different behaviors compete for attention" story.

This remains a **hypothesis, not a confirmed mechanism** (n=1 training run
per module, same caveat as every other finding here). It would predict,
and could be tested by, comparing energy-at-reproduction-threshold
"slack" (excess energy squandered before crossing) between the sparse and
eating-bonus policies, or measuring whether eating-bonus prey/predators
eat measurably more per reproduction cycle than sparse ones do for the
same net progress toward the threshold. Not done here — flagged as the
natural next diagnostic if this line of investigation continues (see
section 8).

**Why kickback still falls short of sparse — credit assignment, not misdirection.**
The argument above explains why kickback beats eating; it does not explain
why kickback still trails pure sparse (86%/96%, not 100%/100%). "Fires on
the same rare event class as the primary reward" is not the same claim as
"reward-neutral" — there is a second, distinct cost specific to kickback,
and it shows up regardless of how well-chosen the event class is.

Look at exactly who gets paid. The mechanism (`agent_parent: Dict[child_id,
parent_id]`, recorded at birth) works like this: when agent X reproduces,
the environment looks up **X's own parent P** — the new grandchild's
grandparent — and pays P the kickback, provided P is still alive. The
triggering event is X's action; the payment goes to P, a different agent
with its own separate trajectory, who did nothing in particular at that
moment. P's only "contribution" was surviving and having had offspring who
themselves went on to reproduce — a real but multi-hop, heavily-delayed,
and largely-out-of-P's-current-control relationship.

Compare this to sparse's reward, which fires exactly when *the same
agent's own* energy crosses *its own* threshold — the tightest possible
credit assignment: the value function only ever has to learn "does my own
current state/behavior predict my own future reward." Kickback keeps that
clean signal fully intact (every agent still gets its own sparse `+10`) but
layers a second income stream on top that the *receiving* agent's own
current policy cannot directly control — it depends on a different agent's
(the child's) independent action, itself only loosely shaped by anything
the parent did much earlier, and further discounted by arriving even later
in the causal chain than the parent's own reproduction did. That is
additional variance in the value-function target with no correspondingly
tight causal handle for the policy gradient to exploit.

This is a different failure mode from eating's, not a smaller instance of
the same one:

- **Eating** pays the agent that performed the action, but the action
  isn't tightly coupled to the *rate* of reaching the true objective —
  it creates a real incentive to do a bit more of it than optimal
  (misdirection).
- **Kickback** pays out only for genuinely on-target events, but to a
  *different* agent than the one whose action caused it — it adds
  attribution noise without ever pointing the policy in a wrong direction
  (mis-attribution, not misdirection).

Both cost something relative to sparse, through different channels — which
is why sparse remains the ceiling: it is the only one of the three with
neither problem. It also explains, structurally, why kickback can approach
sparse but never exceed it: kickback's reward is literally sparse's reward
plus an extra noisy, weakly-attributable term, and adding noise to an
already-maximally-tight objective has no mechanism to help, only to cost a
little within a fixed training budget.

This is the same underlying principle as the credit-assignment argument
against a *learned* nuptial-gift-donation action in the
`eco_evolutionary_nuptial_gift` line of the Darwin/Baldwin evolutionary
project (see that module's README) — reward paid to one agent for a
*different* agent's action is a credit-assignment gap regardless of
whether the underlying event is thematically on-topic. There, the gap was
total (the donor received nothing at all, ruling out a learned action
entirely, hence that module's own donation being modeled as a heritable,
mechanically-executed trait rather than a policy action). Here the gap is
partial and much milder — the grandparent's own sparse reward is untouched,
kickback is a bonus on top — but it's the same structural point in a less
severe form. As with the eating explanation above, this is a plausible,
unconfirmed reading (n=1); a direct test would compare the *variance* of
realized return across kickback-eligible vs. childless agents holding
policy fixed, which hasn't been done here.

## 6. Implications for the Darwin/Baldwin evolutionary project

This whole investigation started as groundwork for the project's evolutionary
trials (see `project_darwin_baldwin_experiment_goal` in memory). The
motivating hypothesis for that groundwork (fix sparsity, then retry) is now
falsified — reward density is not the fix worth pursuing.

## 7. Open questions and caveats

- **n=1 training run per condition.** Every result above comes from a single
  full training run per module (3 evaluation seeds at the end, not 3
  independent training seeds). Unlike the Darwin/Baldwin trials' established
  3-seed training practice, there's no statistical replication here yet —
  treat magnitudes as indicative, not definitive.
- **Findings may be specific to this environment's event frequencies**, not
  universal. The mechanistic explanation in section 5 rests on this
  environment's particular gap sizes (~50-200 steps) and `gamma=0.99`
  effective horizon (~100 steps) being in the same range; environments with
  much larger gaps or different discounting might behave differently.
  Untested here.

## 8. Conclusion

All five modules planned for this investigation are now complete. Ranked by
reproduction-rate recovery relative to the sparse baseline (predator% /
prey%, both against `base_environment_sparse_rewards` = 100%/100%):

1. **Sparse (baseline)** — 100% / 100%
2. **Kickback** (`+10` kin kick-back on grandchild birth) — 86% / 96%
3. **Sparse + eating** (`+1`/`+0.1` asymmetric eating bonus) — 82% / 94%
4. **Dense + additive** (per-step delta + `+10` reproduction bonus) — 63% / 76%
5. **Dense (pure)** (per-step delta only, no reproduction bonus) — 42% / 53%,
   the only variant with an observed extinction event

**The headline result stands, refined rather than overturned.** The
original motivating hypothesis — that the sparse reward was starving
training of signal and that a dense, biologically literal per-step reward
would fix it — is not just unsupported but backwards: every dense variant
underperforms every sparse-family variant, on every axis measured, and pure
dense is the only design to produce an extinction event at a fully-trained
checkpoint. Reward density hurt, it didn't help.

**The refinement kickback adds**: it's not simply "sparse beats dense" or
even "discrete beats continuous" as flatly as section 5 first suggested —
magnitude alone doesn't predict the damage a secondary reward does; firing
*frequency relative to the true objective* does. A shaping term is only
guaranteed policy-invariant if it's potential-based (Ng, Harada & Russell,
1999); a flat per-event bonus for a frequently-occurring intermediate
behavior (eating: ~4-60 times per reproduction cycle) has no such guarantee
and can contribute a large, non-negligible share of total return (up to
~38% for prey) without that share necessarily tracking progress toward the
true objective. Kickback's bonus, though ten times larger per event, only
ever fires on the same rare event class it's meant to reinforce
(reproduction, one hop removed via kinship) — so it costs the least of the
four shaping variants despite being the *largest* in magnitude; eating
(smaller, but far more frequent, and not tied to the terminal event) costs
a bit more; dense (a continuous per-step signal, effectively firing on
every single step) costs the most.

Kickback still trails pure sparse, though (86%/96%, not 100%/100%), and
that gap is a *second*, distinct cost, not a smaller dose of eating's
problem. Eating pays the acting agent, but for a behavior not tightly
coupled to the objective's rate (misdirection). Kickback pays out only on
genuinely on-target events, but to a *different* agent than the one whose
action caused it — the grandparent, for the child's reproduction, an event
the grandparent doesn't control in the moment (mis-attribution, not
misdirection). That distinction is why sparse is the ceiling every variant
approaches from below but none exceeds: kickback's reward is structurally
sparse's reward plus an extra noisy, weakly-attributable term, and noise
added to an already-maximally-tight objective has no mechanism to help,
only to cost a little. The same principle — reward paid to one agent for a
different agent's action being a credit-assignment gap regardless of
topical relevance — is what ruled out a *learned* nuptial-gift-donation
action (as opposed to a heritable, mechanically-executed trait) in the
`eco_evolutionary_nuptial_gift` line of the Darwin/Baldwin project; see
section 5 for the full argument.

**Practical takeaway for this codebase**: when adding a new reward-shaping
mechanism to a `PredPreyGrass` variant (here or in the Darwin/Baldwin
evolutionary line), prefer sparse, discrete, reproduction-topical signals
over continuous per-step ones, and expect a plain `+10`-on-reproduction
baseline to be a genuinely hard bar to clear rather than a naive strawman
worth automatically improving on.

**What would still need testing to firm this up**: a true n≥3 replication
per module (section 7's biggest open caveat), and a kickback-magnitude
sweep to separate "firing-frequency tied to the terminal event" from raw
"magnitude" as competing explanations for kickback's result — both out of
scope for this investigation as currently resourced.

## References

**Reward shaping theory:**
- Ng, A. Y., Harada, D., & Russell, S. (1999). *Policy invariance under
  reward transformations: Theory and application to reward shaping.* In
  Proceedings of the Sixteenth International Conference on Machine
  Learning (ICML 1999) — the potential-based-shaping theorem underpinning
  section 5's explanation for why kickback recovers more of sparse's
  reproduction rate than the eating bonus despite a larger per-event
  magnitude: only shaping terms expressible as `γΦ(s') − Φ(s)` for some
  potential function `Φ` are guaranteed to preserve the optimal policy: a
  flat per-event bonus for a frequently-occurring intermediate behavior
  (eating) has no such guarantee, while kickback's bonus — gated by the
  same rare event class (reproduction) it reinforces — structurally cannot
  be inflated by "doing more of an unrelated frequent behavior" the way an
  eating bonus can.

**Multi-agent credit assignment:**
- Wolpert, D. H., & Tumer, K. (1999). *An Introduction to Collective
  Intelligence.* NASA Ames Research Center, Technical Report NASA-ARC-IC-
  99-63 — introduces difference rewards/utilities: the general principle
  that a reward cleanly reinforces an agent's behavior only to the extent
  it reflects that *same* agent's own contribution, not another agent's
  action. Underpins section 5's explanation for why kickback still trails
  pure sparse: the grandparent is paid for the child's reproduction event,
  not its own current action, adding attribution noise regardless of how
  on-topic the triggering event is.
- Foerster, J., Farquhar, G., Afouras, T., Nardelli, N., & Whiteson, S.
  (2018). *Counterfactual Multi-Agent Policy Gradients.* AAAI 2018 (COMA)
  — the modern deep-RL instantiation of the same credit-assignment
  principle, addressing exactly this failure mode (an agent's reward
  entangled with other agents' actions makes it hard for policy gradients
  to learn what that agent's *own* actions are worth) in a multi-agent
  setting structurally similar to this environment's shared per-species
  policies.
