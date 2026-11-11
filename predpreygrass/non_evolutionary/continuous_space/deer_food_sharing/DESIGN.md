# Design note: sharing big game (2026-11-11)

Status: **approved 2026-11-11** (the user followed the recommended proposal);
pre-registered in RESULTS.md section 1 before any training run.

## Why

In deer_stag_hunt (touch rule, its RESULTS.md section 2), hunters that only learn
approach adult deer with a partner near slightly more often than chance (1.12-1.21),
but adults bring the hunters only a few percent of their energy. In
eco_evolutionary_deer_stag_hunt section 3, peeks at the main runs suggest that an
evolving innate bias steers hunters to young deer, which raises births but lowers
partnering on adults (not yet tested; that stage's verdict is due 2026-11-13).

In human foragers the big, unpredictable packages are the ones shared widely, small
items much less (Kaplan and Hill's work on food sharing). Sharing pools the risk of a
failed hunt, and is one of the main proposed reasons big-game hunting pays at all. The
deer world has no sharing: each hunter keeps its own catch. This module adds it, and
nothing else.

## The rule

When an **adult** deer is caught:

- every other living hunter within **R** of the deer, from the positions at the
  start of the physics step, is a **bystander**, whether it helped or not ("tolerated
  scrounging");
- if there is at least one bystander, the catchers split (1 − **s**) of the food
  (the catch efficiency × the deer's energy) evenly and the bystanders split **s**
  of it evenly; without bystanders the catchers keep it all, as without sharing.

**Fawns and middle deer are never shared** beyond their catchers. Observations,
actions, rewards and every other setting are deer_stag_hunt's touch world (its
RESULTS.md section 2, instant grass, `deer_young_solo` "graded").

**Values, fixed now:** s = 0.5, R = 128 (twice the party radius). One value of each,
no tuning.

**Why this can matter, and why it might not:**

- *Insurance:* a hunter who failed still eats from a neighbour's adult, which lowers
  the risk of trying for adults.
- *Kin:* offspring are born next to their parent, so bystanders tend to be relatives;
  sharing partly feeds kin without kin recognition being built in.
- *Free-riding:* catchers get less per adult, and following other hunters without
  hunting is rewarded. Learning may find that instead.
- *Payoffs:* adults are a few percent of the hunters' energy; sharing moves that
  energy around but does not add any. Sharing alone may change nothing.

## Comparison

Three runs with sharing (share_1-3), against deer_stag_hunt's touch_1-3, which are the
same world without sharing. Same code apart from the rule (with s = 0 this module's
world is deer_stag_hunt's, episode for episode, tested), same PPO settings, 1000
iterations, episodes drawn from the operating system's entropy as there. Measured at
the final checkpoints with the same evaluation (evaluate.py 50 episodes, seeds 0-49;
coop_fair.py 20 episodes); this module's evaluation reproduces touch_1's evaluation
exactly (all 50 episodes, checked 2026-11-11). touch_1-3 are historical controls: trained earlier, with
their values known when the tests were set, so the comparison is a pre-registered
ordering criterion, not a randomized experiment (RESULTS.md section 1.3).

Tests and the hard stop: RESULTS.md section 1.

## Not done here

- No genes. If the sharing runs show higher partnering on adults here, an evolutionary version
  (B against L with sharing) is a separate, later decision.
- Kin are not tracked; whether bystanders are relatives is not measured.
