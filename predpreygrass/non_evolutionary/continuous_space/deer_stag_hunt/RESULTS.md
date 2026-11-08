# deer_stag_hunt: results

## 1. Replication without calls (2026-11-07)

**Question:** does the result of [`stag_hunt_continuous`](https://github.com/doesburg11/PredPreyGrass-archive/blob/main/stag_hunt_continuous/RESULTS.md)'s RESULTS.md section 8 (archived)
(coexistence and adult approaches with a partner near above chance) hold without
hunters' calls, and in more than one training run? Section 8 was a single run, with
calls and the catching-together rule added at once; silencing the calls of its trained
hunters changed nothing, but whether calls mattered during training was not tested.

**World:** this module's defaults (README.md): section 8's world without calls. The
predators' network also loses section 8's call output and its always-zero inputs
(README.md, Origin), a second difference besides the calls.

**Runs:** three independent runs from scratch (`run_1`, `run_2`, `run_3`), 1000
iterations each, 14 env runners, `launchers/replicate.py`. Runs differ only by chance
(network initialisation and episode draws). Two may train at once if the GPU has room;
the evaluations are the same either way.

**Tests, set before training** (each run's final checkpoint; as section 8's tests 1-3):

1. **Coexistence:** hunters and deer both alive at the end of at least 45 of 50
   evaluated episodes (`evaluate.py`, 50 episodes).
2. **Cooperation:** with the fair measure (`launchers/coop_fair.py`, 20 episodes),
   approaches to adults have another hunter within 64 units more often than approaches
   to fawns and than chance.
3. **Above section 7:** the adult share in test 2 divided by chance above 0.90, the
   value of `stag_hunt_continuous`'s section 7 (lean and fat rabbits, a fat rabbit
   catchable by a party within 64, no touching rule).

**Verdict, set before training:** replicated if all three tests are met in at least 2
of the 3 runs; not replicated otherwise. Each run's result is reported either way.
`launchers/verdict.py` applies the tests and the verdict to
`results/run_<n>_{eval,coop}.json`.

Also recorded, not tests: iterations 250 and 500; the bootstrap 95% interval over
episodes of adult share minus chance (`coop_fair.py`, 5000 resamples); the adult energy share and catch counts;
section 8's final values for comparison (49 of 50; adult share ÷ chance 1.18).

**Hard stop:** no change to the world during or after these runs; what follows is the
user's decision.

**Caveats, set in advance:** three runs show run-to-run variation but do not make a
population estimate. A replication would show that calls (and the inputs and output
removed with them) were not needed for this result in this world; a failure could mean calls helped in section 8 or that section
8's run was lucky, and these runs could not tell which. The rule makes a partner
necessary for an adult, so what is tested is coordination where the task requires two:
a lone hunter can still live on fawns.

### 1.1 Results (2026-11-08)

**Verdict: replicated.** `launchers/verdict.py`: run_2 and run_3 meet all three
tests; run_1 meets tests 2 and 3 but misses test 1 (43 of 50).

Final checkpoints (iteration 1000), the tests:

| Run | Coexisting (of 50) | Chance | Fawn share | Adult share | Adult ÷ chance | Adult − chance, 95% interval | Tests met |
| --- | --- | --- | --- | --- | --- | --- | --- |
| run_1 | 43 (7 with hunters extinct) | 0.074 | 0.058 | 0.089 | 1.20 | [0.002, 0.028] | 2, 3 |
| run_2 | 49 (1 with hunters extinct) | 0.155 | 0.117 | 0.190 | 1.23 | [0.024, 0.045] | 1, 2, 3 |
| run_3 | 50 | 0.125 | 0.095 | 0.144 | 1.15 | [0.010, 0.027] | 1, 2, 3 |
| section 8 (with calls, one run) | 49 | 0.132 | | 0.155 | 1.18 | | |

Also recorded (not tests):

| Run | Checkpoint | Coexisting | Adult ÷ chance | Adult − chance, 95% interval | Episodes adult > chance (of 20) | Adult energy share | Adults caught (50 episodes) | Hunter births (50 episodes) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run_1 | 250 | 50 | 1.25 | [0.035, 0.053] | 19 | 0.101 | 1757 | 1143 |
| | 500 | 50 | 1.20 | [0.020, 0.049] | 17 | 0.089 | 1481 | 1042 |
| | 1000 | 43 | 1.20 | [0.002, 0.028] | 11 | 0.030 | 243 | 439 |
| run_2 | 250 | 50 | 1.19 | [0.018, 0.032] | 20 | 0.064 | 1043 | 1106 |
| | 500 | 50 | 1.18 | [0.019, 0.036] | 18 | 0.079 | 1298 | 1046 |
| | 1000 | 49 | 1.23 | [0.024, 0.045] | 18 | 0.075 | 1130 | 920 |
| run_3 | 250 | 50 | 1.21 | [0.024, 0.047] | 18 | 0.082 | 1393 | 1141 |
| | 500 | 50 | 1.17 | [0.018, 0.044] | 18 | 0.090 | 1491 | 1071 |
| | 1000 | 50 | 1.15 | [0.010, 0.027] | 16 | 0.055 | 706 | 740 |

**What this shows.** Section 8's result held without calls, in all three runs at
iterations 250 and 500 (50 of 50; adult ÷ chance 1.17-1.25, intervals above zero) and
in two of three at iteration 1000. Within this world, calls (and the inputs and output
removed with them) were not needed for coexistence and for adult approaches with a
partner near above fawns and chance.

**Limits.**

- **The effect is small.** At iteration 1000, adult approaches have a partner near 15-23% more often
  than chance, and adults give 3-10% of hunters' energy; most food is still fawns and
  middle deer caught alone.
- **Late decline in two runs.** From iteration 500 to 1000, adult catches fell by
  84% in run_1 (1481 to 243; 7 episodes lost all hunters) and by 53% in run_3 (1491
  to 706), hunter births with them; run_2 fell by 13%. In run_3, the training curves
  (smoothed, with exploration) were still rising at iteration 785, so the drop shows in
  evaluation, not in training. Why is not known; this experiment did not test it. The
  pre-registered tests are on iteration 1000 and stand as reported; on iteration 500,
  all three runs would have met all tests, but that was not the test.
- **Approach with a partner near is the measure, not proof of seeking a partner:**
  hunters may meet near adults for other reasons (shared grass areas, deer herds), which
  the chance baseline only partly removes.
- **The fawn comparison is partly built in.** A fawn or middle deer touched with another
  hunter within 64 units is shared with every hunter that near, so a partner near a
  fawn halves the meal. Hunters therefore have a reason to avoid others near fawns,
  which lowers the fawn share in test 2. The chance comparison does not have this
  bias (adults were above chance in all runs), but part of the adult-fawn gap may
  reflect avoiding sharing rather than seeking partners. (Added 2026-11-08, after
  the verdict; it does not change the pre-registered tests.)
- Three runs show run-to-run variation; they are not a population estimate.

**Hard stop reached.** No world changes followed; the next step is the user's decision.

## 2. One touch rule for every deer (pre-registered 2026-11-08)

**Question:** does section 1's result hold in a world where only touching counts?
In section 1's world, a young deer touched while another hunter was within 64
units was always caught and shared by every hunter that near, without the others
touching it. That rule is not needed for the stag hunt, gives nearby hunters a share
for nothing, and builds in part of the fawn comparison (section 1.1, limits).

**World:** section 1's world with `deer_catch_rule: "touch"` (environment.py
docstring), nothing else changed:

- every touch is a solo attempt, caught with chance 1 for a fawn, falling linearly
  from 1 at energy 2 to 0 at energy 5, and 0 for an adult; a caught deer goes to the
  toucher alone;
- a failed attempt costs the toucher a wait of 8 physics steps, as before;
- once two or more hunters have touched a deer within 8 physics steps (at once, or
  after a failed attempt), it is caught whatever its size and shared by those
  touchers only.

Adults are caught as in section 1. Per meal, a partner pays for deer above energy
3.5 (README.md, The world); the payoffs are section 1's.

**Runs:** three runs from scratch (`touch_1`, `touch_2`, `touch_3`), 1000 iterations,
14 env runners, `launchers/replicate.py NAME --catch-rule touch`, two at a time.

**Tests and verdict:** section 1's, unchanged (coexistence at least 45 of 50;
adult share above fawn share and chance; adult ÷ chance above 0.90; all three in at
least 2 of 3 runs), on the final checkpoints, with `launchers/verdict.py touch_1 touch_2
touch_3`. The cooperation measure is unchanged: a partner near means another hunter
within 64 units of the deer at the approach, although under this rule such a hunter
gets no share unless it touches.

**Also recorded, not tests:** iterations 250 and 500; the share of young catches
made by two touchers; adults caught and hunter births.

**Hard stop:** no setting changes during or after these runs, whatever the outcome.
What follows is the user's decision.

**What outcomes would mean:** a replication would show section 1's result does not
need the party rule for young deer. A failure would mean it did need it in this
world, through coexistence or through cooperation; three runs could not tell more.

### 2.1 Partnering by the deer's energy (pre-registered 2026-11-08, before the final checkpoints)

**Question:** under the touch rule, a middle deer (energy between 2 and 5) can be
tried alone (chance (5 − energy) / 3) or caught for certain by two touchers. Per
hunter, alone pays more below energy 3.5 and a partner pays more above it (README.md,
The world). No rule forces a partner, so this is the stag hunt in its strict sense,
which no earlier world of this line showed. Do hunters approach the larger middle
deer with a partner near more often, and above chance?

**How this was set (exploratory origin, stated in full):** the question came up after
section 2's pre-registration. A first look on the interim checkpoints touch_1 iteration
390 and touch_2 iteration 350, and on section 1's run_1-3 at iteration 500 (20 episodes
each), showed partnering rising with energy under the touch rule, reaching chance in the
top middle bin, but not under the party rule. The tests below were then set, before
any final checkpoint exists. The interim checkpoints enter no test; touch_3 was not
looked at.

**Measure:** `launchers/coop_by_energy.py`. Approaches as in `coop_fair.py` (its counts
are reproduced exactly), binned by the deer's energy at the approach. Near: another
hunter within 64 units of the deer, as in section 1. Chance: as in section 1. Also
tested from the amendment below: joint touch (another hunter ready to catch, not in a
catch cooldown, touched the same deer within 8 physics steps of the approach, before or
after). Under the touch rule that is what catches a deer together; a hunter merely
near gets nothing, so near is the measure kept from section 1 and joint the one that
fits this rule.

**Tests,** on each of touch_1-3's final checkpoints, 40 episodes (seeds 0-39), 95%
bootstrap intervals over episodes:

- **A (rise):** near share for middle deer with energy in (4, 5) minus near share for
  energy in (2, 3.5]; the interval above 0.
- **B (above chance where a partner pays):** near share for middle deer with energy in
  (4.5, 5) minus chance; the interval above 0.
- **C (teaming up where a partner pays, as for adults):** joint share for middle deer
  with energy in (4, 5) divided by joint share for adults; the interval above 0.5.
  Adults must be caught by two, so their joint share is the level of teaming where it
  is required.

**Verdict:** the pattern is shown if A, B and C are all met in at least 2 of the 3 runs
(`launchers/energy_verdict.py touch_1 touch_2 touch_3`).

**How a positive verdict may be described:** as a hypothesis from interim checkpoints
confirmed on the final ones, not as an independent confirmation: touch_1 and touch_2
informed the tests (at earlier checkpoints) and can make the verdict. touch_3, the
only run not looked at, is reported on its own as well.

**Comparison, no test:** section 1's run_1-3 final checkpoints (party rule) and
touch_1-3 at iteration 500, same measure. In the first look, test A was met under the
party rule too (all three runs at iteration 500), so A alone does not separate the
rules; B did (party-rule values −0.028 to −0.030, intervals below 0). A is kept as a
necessary condition.

**Amendment (2026-11-08, 20:36, still before any final checkpoint): test C added.**
Near has no part in catching under the touch rule, so a test on joint touches was
added. Its form was chosen on the same first-look data: a rise in joint share from
(2, 3.5] to (4, 5) was met in 2 of 3 party-rule runs as well, so it would not separate
the rules; the ratio to adults did: 0.89 in both touch checkpoints (intervals
[0.78, 1.00] and [0.78, 1.04]), 0.18 to 0.25 in the party-rule runs. The threshold
0.5 was set between those values, which makes C, like B, informed by the first look.

**Limits, set now:** big middle deer may share places with adults, so a met B is
consistent with partnering by payoff but also with place or with generalising from
adults; it is not proof of choosing a partner by payoff. The crossing in the first
look was near energy 4.5, not at the payoff break-even 3.5. A control world (adults
catchable alone) would separate place from cooperation; it is not part of this
section.

**Running:** `launchers/run_energy.sh` waits for section 2's runs and evaluations, then
runs all of the above. Results to `results/<run>_energy.json` and
`results/energy_verdict.txt`. No setting changes; hard stop as in section 2.
