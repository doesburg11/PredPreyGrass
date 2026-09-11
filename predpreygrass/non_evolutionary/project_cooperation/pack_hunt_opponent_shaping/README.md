# Pack Hunt Opponent Shaping

**Status: all three pieces exist and are verified -- environment, naive-PPO
baseline (condition 1), and pairwise N-player opponent-shaping (condition 2).
No full training run to convergence has been done yet; what's verified is
correctness of the mechanism, not the resulting equilibrium behavior.**

`predpreygrass_rllib_env.py` implements the mechanics below as an RLlib
`MultiAgentEnv`; `random_policy.py` runs it with uniformly random predator
actions and a pygame viewer (`utils/pygame_renderer.py`) so the engagement/
effort-cost/catch mechanic can be watched directly; `tune_ppo.py` trains it
with independent per-predator RLlib PPO and no opponent-awareness (condition
1); `tune_opponent_shaping.py` (`opponent_shaping/`) is the pairwise N-player
opponent-shaping training loop (condition 2) -- a from-scratch policy-
gradient loop, not RLlib, for the reason Section 8 gives.

That opponent-shaping implementation was checked the same way this whole
project family checks this kind of thing -- not trusted on inspection alone:
the per-sample score function (needed because a neural-network policy has no
closed-form score the way Foerster2018's 5-parameter table does) is verified
against brute-force `torch.autograd.grad`; the pairwise correction's
Hessian-vector-product formulation (needed because materializing a full
parameter x parameter matrix per pair is impractical at network scale) is
verified against the naive materialized-matrix computation for exact
numerical equivalence; and, strongest of all, **the N-player update is
verified to reduce exactly to Foerster2018's own `lola_pg_update` function,
called directly from the sibling reproduction repo, at N=2** -- this is a
generalization of that reproduction, not a loose reinterpretation inspired
by it.

**Independent review.** Per this project family's standing practice, `codex
exec` (OpenAI Codex CLI) reviewed `utils/policy_network.py`,
`opponent_shaping/pairwise_lola_pg.py`, `opponent_shaping/rollout.py`, and
`tune_opponent_shaping.py` for correctness bugs specific to N >= 3 (i.e.
that wouldn't show up in the N=2-vs-Foerster2018 comparison above), rollout
timestep/indexing bugs, and autograd/state-leakage issues. It confirmed the
pairwise sum, timestep alignment, and per-batch environment independence
were all correct, and found one real gap plus three low-severity portability
issues, all fixed: `collect_rollout` didn't guard against a rollout horizon
exceeding `max_episode_steps` (would silently step an already-truncated
env rather than reset, splicing two logical episodes into one trajectory --
now raises); the math utilities and rollout storage hardcoded CPU/float32
instead of deriving device/dtype from the model (now fixed, for future
GPU use); and per-sample scores retained an unnecessary autograd graph
(now explicitly detached, matching Foerster2018's own "scores are plain
numbers" convention). All four correctness checks above were re-run after
the fixes and still pass exactly.

This README documents the design as worked out in full before the code was
written, in the same spirit as this repo's other `research_question.md`
notes: the reasoning and the rejected alternatives are kept, not just the
final answer, so the choices can be revisited if the resulting dynamics
don't look right.

Run the viewer with:
```
python -m predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.random_policy
```

Run the naive-PPO baseline with:
```
python -m predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.tune_ppo
```

Run the pairwise opponent-shaping loop with:
```
python -m predpreygrass.non_evolutionary.project_cooperation.pack_hunt_opponent_shaping.tune_opponent_shaping
```

## 1. Where this comes from

This module grew out of a question about
[Foerster et al. (2018)](https://github.com/doesburg11/Foerster2018)'s LOLA
(Learning with Opponent-Learning Awareness): that paper's exact- and
policy-gradient methods are strictly **two-player**, built and validated only on
the Iterated Prisoner's Dilemma and Iterated Matching Pennies. The question was
whether the same idea — an agent that differentiates its own update through its
opponent's anticipated learning step, instead of treating the opponent as a fixed
part of the environment — extends to more players, and specifically whether it
says anything useful about cooperation in a PredPreyGrass-style multi-agent
ecology.

`Foerster2018` itself stays a from-scratch **reproduction repo only** (two-player
matrix games, nothing else) by explicit project policy, so this module lives here
instead, as an original, LOLA-*inspired* extension — not a reproduction of
anything in the paper.

## 2. The reframing that made this well-posed: predators aren't cooperating with prey

The first framing to reject: predator-vs-prey is not a cooperation relationship at
all. They are pure opponents with no mutually-better joint outcome to shape toward
— there's nothing for opponent-shaping to do between a predator and its meal. The
actual social dilemma is **intra-group**: do multiple predators sharing a hunt
coordinate effort, or does each individually try to free-ride on the others'
effort? (The symmetric question — do prey cooperate on vigilance/evasion, or act
selfishly — is a natural extension but explicitly out of scope for this first
version; predators only.)

## 3. Why this needs its own fixed-population module, not a variant of `stag_hunt_*`

`stag_hunt`, `stag_hunt_defection`, `stag_hunt_reputation`, `mammoths`, and
`mammoths_defection` already implement join/free-ride and reputation-conditioned
cooperation mechanics that look superficially similar to what's described below.
They are not a fit for this experiment, for a structural reason: they all run on
top of the **full ecology** — energy decay, reproduction, spawning and dying
predators and prey, with prey also learning. That's an *evolving population*, and
LOLA-style opponent shaping doesn't have a well-defined meaning there:

- Its whole mechanism is "differentiate through *this specific opponent's* next
  gradient update." If that opponent can die and be replaced, or a new agent can
  spawn mid-training, "anticipate their next update" stops being a coherent
  target — the population underneath the training run isn't stable across the
  timescale the correction term is reasoning about.
- The end goal here is to isolate one clean question — does opponent-shaping
  change the intra-predator cooperate/free-ride equilibrium — without a second,
  confounding process (who's even alive to cooperate with) running at the same
  time.

So this module deliberately **removes** the things every sibling module in
`project_cooperation/` treats as core: no reproduction, no death, a fixed
predator population for the whole run, and prey that are **scripted, not
learning** (simple evasion heuristic, not a trained policy). This trades away
ecological realism on purpose, in exchange for a well-posed multi-agent learning
question with a stable set of co-learners — the same fixed-population assumption
LOLA's own two-player experiments rely on.

Also deliberately excluded for the same reason: no metabolic energy/starvation
mechanic. Every sibling module models energy decay and death from starvation;
this one doesn't. Adding it back would reintroduce a second, confounding
dilemma (a survival-pressure decision layered on top of the cooperation
decision), muddying whatever the opponent-shaping comparison actually shows. If
this module's dynamics turn out to be worth trusting, reintroducing energy decay
as a deliberate follow-up — to see whether the cooperation pattern found here
survives real starvation risk — is a natural next step, just not this one.

## 4. The stage-game dilemma: producer–scrounger, not stag-hunt-style coordination risk

Two candidate dilemma structures were considered for "what's the actual downside
of cooperating":

- **Coordination risk (stag-hunt structure)**: commit to a joint strategy that
  only pays off if a partner also commits; if the partner doesn't show up, the
  committed agent gets nothing. This is the mechanic
  [Leibo et al. (2017)](https://humanbehaviorpatterns.org/learned-cooperation/leibo2017)'s Wolfpack environment
  actually uses (already reproduced in this project's sibling
  [Leibo2017](https://github.com/doesburg11/Leibo2017) repo).
- **Effort cost (producer–scrounger / public-goods structure)**: pursuing is a
  guaranteed private cost paid regardless of outcome; the reward, when it
  happens, is shared with anyone nearby regardless of whether they paid that
  cost.

The producer–scrounger structure was chosen as more directly biologically
grounded: Packer & Ruttan (1988) and Packer & Pusey (1997) document exactly this
pattern in real lion prides — some individuals ("laggards") consistently let
others do the costly, risky work of driving and cornering prey, then share in
the kill regardless, and are not retaliated against for it. Giraldeau & Caraco's
producer–scrounger game (2000) is the general behavioral-ecology framework for
this: a private-cost/shared-benefit dilemma, not an all-or-nothing bet on a
partner. Real hunts degrade gracefully as fewer individuals pay the effort cost
(lower success probability), rather than failing outright the way a stag-hunt
framing implies — that graceful-degradation property is exactly what "catch
probability increases with number of engaged predators" (below) is meant to
capture.

## 5. What counts as "engaging" — proximity, not an action or a movement direction

Two alternatives were rejected before settling on this:

- **An explicit Pursue/Scrounge action, chosen independently of movement**: this
  is cheap talk — an agent could declare "Pursue" while moving away from the
  prey and still collect cooperation credit, unless heavily gated.
- **Inferring engagement from movement direction** (e.g. "reduced distance to
  prey this step = pursuing"): this breaks down for any real coordinated
  cornering/flanking maneuver, where a predator looping around to block an
  escape route can be *increasing* raw distance to the prey while doing
  genuinely costly, useful work. There's no clean rule for "the right direction"
  once coordination requires anything beyond charging straight at the target.

**Resolution: engagement is a derived state based on physical proximity, not a
choice.** There is no separate Pursue/Scrounge action — agents only ever choose
ordinary movement. Each round, whichever predators end up within a small
**engagement radius** of the prey are automatically classified as *engaged* that
round: they pay a fixed cost `−c` and contribute to that round's catch
probability. Predators outside the radius are *disengaged*: no cost, no
contribution. This ties the cost directly to the real physical source of risk
(proximity to a fleeing/defensive animal), with no directional judgment call and
no cheap-talk gap.

### The three radii

| Radius | Role | Starting value (10×10 grid, 1 cell/step) |
|---|---|---|
| Capture radius | Actual tag/catch check | 1 (adjacent) |
| Engagement radius | Pays `−c`, contributes to catch probability | ≈ 2 |
| Reward-sharing radius | Must be within this at the tag to collect a share | ≈ 2–3 (≥ engagement radius) |

These are a starting hypothesis, not derived constants — there's no first-principles
value here, only a tradeoff to sit inside: too small an engagement radius and
"engaged" collapses into "present at the capture instant" (scrounging becomes a
free last-second dash); too large and nearly every predator counts as engaged
almost all the time, turning `−c` into a constant tax rather than a real choice.
The sharing radius is set `≥` the engagement radius on purpose, so a scrounger has
an actual niche: loiter just outside engagement, pay nothing, still be close
enough to collect a share when a catch happens.

### Reward split

If a round ends in a catch, the reward is split **equally among every predator
within the sharing radius, engaged or not** — no bonus for having paid the
effort cost. This is deliberate, not an oversight: it's what makes scrounging
actually tempting rather than obviously dominated, and it matches the empirical
lion/wolf "laggard" pattern directly (Section 4) rather than a scheme where
free-riding is trivially punished by the reward function itself.

## 6. Rounds, not single-shot episodes

An early version of this design ended the episode at first capture. That's a
mistake worth naming: it collapses the whole cooperate/defect question to a
**single** moment per episode, with nothing for a policy to condition on — no
memory-one-style reciprocity is even expressible if there's no "last round" to
react to.

**Fix: a capture (or a timed-out failed hunt) ends a *round*, not the episode.**
The prey respawns and the same fixed group of predators immediately faces
another hunt. This restores genuine within-episode repetition — the same
structural role IPD's repeated C/D rounds play for LOLA — so reciprocity has
something to be conditioned on.

### Per-round timeout is load-bearing, not just tidy

Each round has a max-steps cap. This isn't primarily about incentivizing
individual predators to "do something" — scrounging is a legitimate strategic
choice, not an error. It exists because **without a cap, universal scrounging
(zero engaged predators) gives a catch probability of exactly zero, and the round
never ends at all** — the exact same failure mode as NL-vs-NL's permanent mutual
defection in IPD, except unbounded and unobservable instead of a clean, finite,
informative outcome. The timeout turns "the group collectively failed to
engage" into a real, scoreable event (≈0 reward, next round starts) — the
moral equivalent of IPD's `(-2,-2)` mutual-defection outcome — rather than a
stalled simulation.

### Memory: the minimum needed for reciprocity

Each predator observes, from the *previous round only*, which peers were
engaged and how that round ended — the direct analog of IPD's `CC/CD/DC/DD`
memory-one states. This is the minimum sufficient statistic for a TFT-like
policy to exist at all: *join the hunt if my podmates joined last round; hang
back if they free-rode on me last round.*

## 7. Discounting, not a fixed known round count

A fixed, known total number of rounds per episode was considered and rejected.
With `γ = 1` and a known horizon, the last round has no future to protect, so
free-riding is unpunishable there — and by backward induction that unravels
earlier rounds too. This isn't hypothetical: this project's own PPO case study
on the plain repeated Prisoner's Dilemma (`n_rounds = 50`, fixed and known)
converges to all-defect, which is exactly the backward-induction signature.

**Resolution: no fixed known round count.** Use `γ < 1` over an open-ended
sequence of rounds (equivalently, a per-round continuation probability), so no
round ever looks like "the known last one" from inside the game. This mirrors
exactly how `Foerster2018`'s own exact-gradient IPD/IMP experiments are
structured — no stated horizon at all, just `V = Σ γ^t r_t` in closed form, with
the discount factor standing in for "the game probably continues." `γ` isn't a
cosmetic hyperparameter here: it's what determines whether the memory/reciprocity
mechanism in Section 6 is worth anything at all to a learner. A sufficiently
myopic agent has no reason to act on last round's engagement record, since doing
so only pays off in rounds it's discounting away — and it also modulates how much
there is for the opponent-shaping correction term itself to anticipate (Section
8), since a short effective horizon shrinks the future that term is reasoning
about.

## 8. Learning algorithm

Three conditions to compare:

1. **Naive** — independent policy-gradient, no opponent-awareness. Expected
   baseline: free-riding dominates, low engagement rate.
2. **Pairwise N-player opponent shaping** (implemented, `opponent_shaping/` +
   `tune_opponent_shaping.py`) — each predator's update includes an
   opponent-shaping correction term per *other* predator (summed pairwise),
   rather than the exact 2-player method's exhaustive joint state-space
   enumeration, which scales as `(actions)^N` and is infeasible past a couple
   of agents. This is the direct N-player generalization of LOLA's own
   mechanism, and is checked to reduce exactly to Foerster2018's own
   `lola_pg_update` at `N = 2` -- see `opponent_shaping/pairwise_lola_pg.py`'s
   module docstring for the correctness checks. Because a neural-network
   policy has no closed-form score function the way a 5-parameter table
   does, this needed two departures from Foerster2018's own implementation,
   both purely about *how* the same quantity is computed, not what it is:
   per-sample scores come from `torch.func.grad` + `vmap` instead of a
   closed-form formula, and the pairwise correction is computed as a
   Hessian-*vector* product instead of materializing a full
   parameter x parameter matrix per pair (impractical at network scale, unlike
   Foerster2018's `P=5`).
3. **M-FOS-style meta-policy** (fallback, not built) — if condition 2 proves
   unstable at `N = 3`, fall back to a model-free meta-policy trained via
   ordinary PPO instead of tuning the differentiable pairwise correction
   further.

### Why pairwise opponent-shaping over a fully model-free approach here specifically

M-FOS's main selling point is removing the need for white-box access to an
opponent's actual learning process — important when the two sides are
fundamentally different populations (e.g. predator vs. prey, different
objectives, likely different architectures, no realistic way to see inside each
other). That objection is much weaker for *this* dilemma: the peers being shaped
are same-species predators, co-trained in the same simulation by the same
experimenter. Granting them access to each other's parameters for research
purposes is a reasonable assumption here in a way it wouldn't be across species.

### Relevant follow-up literature (surveyed before choosing method 2)

- Foerster et al. (2018), *DiCE: The Infinitely Differentiable Monte Carlo
  Estimator*, ICML 2018 — the correct way to get LOLA's second-order term from
  plain autodiff instead of hand-derived score-function bookkeeping.
- Letcher, Foerster, Balduzzi, Rocktäschel & Whiteson, *Stable Opponent Shaping
  in Differentiable Games*, [arXiv:1811.08469](https://arxiv.org/abs/1811.08469)
  — fixes LOLA's convergence failures in some games.
- Willi, Treutlein, Letcher & Foerster, *COLA: Consistent Learning with
  Opponent-Learning Awareness*,
  [arXiv:2203.04098](https://arxiv.org/abs/2203.04098) — fixes an internal
  inconsistency in how LOLA models an opponent who is also shaping.
- Zhao, Zhu, Foerster et al., *Proximal Learning With Opponent-Learning
  Awareness (POLA)*, [arXiv:2210.10125](https://arxiv.org/abs/2210.10125) —
  fixes LOLA's sensitivity to neural-network policy parameterization.
- Lu, Willi, Letcher & Foerster, *Model-Free Opponent Shaping (M-FOS)*,
  [arXiv:2205.01447](https://arxiv.org/abs/2205.01447) — the fallback in
  condition 3.
- Balduzzi, Racanière, Martens, Foerster, Tuyls & Graepel, *The Mechanics of
  n-Player Differentiable Games*,
  [arXiv:1802.05642](https://arxiv.org/abs/1802.05642) — the general n-player
  theory (potential/Hamiltonian decomposition, Symplectic Gradient Adjustment)
  this whole line of work sits inside.
- Kim et al., *A Policy Gradient Algorithm for Learning to Learn in Multiagent
  Reinforcement Learning (Meta-MAPG)*,
  [arXiv:2011.00382](https://arxiv.org/abs/2011.00382), ICML 2021.
- Souly, Willi, Khan, Kirk et al., *Leading the Pack: N-player Opponent
  Shaping*, [arXiv:2312.12564](https://arxiv.org/abs/2312.12564), NeurIPS 2023 —
  the most directly applicable prior work: explicitly extends opponent shaping
  from 2 to 3–5 players and reports that its advantage over naive learning
  *shrinks* as the number of co-players grows. That's a useful, concrete caution
  for how far this module's approach could scale toward a larger pack.

## 9. What to measure

- Steady-state engagement rate per predator (do all three converge to engaging,
  does one specialize as a permanent scrounger, does it cycle?).
- Per-agent reward variance (a persistent scrounger should show a different
  variance signature than a persistent engager).
- Round-to-round history dependence: does a predator that scrounged last round
  get excluded from the pack's effective hunting position next round — the
  behavioral signature of TFT-style punishment, which the reward function here
  doesn't hand-code (sharing is automatic-by-proximity, not a voluntary
  transfer, so any exclusion has to be spatial and learned).
- Catch rate per round and engagement rate, compared across the three learning
  conditions in Section 8.

## References

- Foerster, J., Chen, R. Y., Al-Shedivat, M., Whiteson, S., Abbeel, P., &
  Mordatch, I. (2018). "Learning with Opponent-Learning Awareness." *AAMAS 2018*.
- Foerster, J., Farquhar, G., Al-Shedivat, M., Rocktäschel, T., Xing, E., &
  Whiteson, S. (2018). "DiCE: The Infinitely Differentiable Monte Carlo
  Estimator." *ICML 2018*.
- Letcher, A., Foerster, J., Balduzzi, D., Rocktäschel, T., & Whiteson, S.
  (2019). "Stable Opponent Shaping in Differentiable Games." *ICLR 2019*.
  arXiv:1811.08469.
- Willi, T., Treutlein, J., Letcher, A., & Foerster, J. (2022). "COLA:
  Consistent Learning with Opponent-Learning Awareness." *ICML 2022*.
  arXiv:2203.04098.
- Zhao, S., Zhu, C., Grosse, R., & Foerster, J. (2022). "Proximal Learning With
  Opponent-Learning Awareness." *NeurIPS 2022*. arXiv:2210.10125.
- Lu, C., Willi, T., Letcher, A., & Foerster, J. (2022). "Model-Free Opponent
  Shaping." *ICML 2022*. arXiv:2205.01447.
- Balduzzi, D., Racanière, S., Martens, J., Foerster, J., Tuyls, K., & Graepel,
  T. (2018). "The Mechanics of n-Player Differentiable Games." *ICML 2018*.
  arXiv:1802.05642.
- Kim, D. K., et al. (2021). "A Policy Gradient Algorithm for Learning to Learn
  in Multiagent Reinforcement Learning." *ICML 2021*. arXiv:2011.00382.
- Souly, A., Willi, T., Khan, A., Kirk, R., et al. (2023). "Leading the Pack:
  N-player Opponent Shaping." *NeurIPS 2023*. arXiv:2312.12564.
- Packer, C., & Ruttan, L. (1988). "The Evolution of Cooperative Hunting." *The
  American Naturalist*, 132(2), 159–198.
- Packer, C., & Pusey, A. E. (1997). "Divided We Fall: Cooperation among
  Lions." *Scientific American*, 276(5), 52–59.
- Giraldeau, L.-A., & Caraco, T. (2000). "Social Foraging Theory." Princeton
  University Press. (Producer–scrounger game framework.)
- Leibo, J. Z., Zambaldi, V., Lanctot, M., Marecki, J., & Graepel, T. (2017).
  "Multi-agent Reinforcement Learning in Sequential Social Dilemmas." *AAMAS
  2017*. (Wolfpack's coordination-risk mechanic, considered and set aside in
  Section 4.)
