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

### 2.2 Results (2026-11-09)

**Section 2 verdict: replicated.** `launchers/verdict.py touch_1 touch_2 touch_3`: all
three runs meet all three tests (needed: 2 of 3).

Final checkpoints (iteration 1000), the tests:

| Run | Coexisting (of 50) | Chance | Fawn share | Adult share | Adult ÷ chance | Adult − chance, 95% interval | Tests met |
| --- | --- | --- | --- | --- | --- | --- | --- |
| touch_1 | 50 | 0.132 | 0.093 | 0.148 | 1.12 | [0.006, 0.026] | 1, 2, 3 |
| touch_2 | 50 | 0.142 | 0.099 | 0.171 | 1.21 | [0.023, 0.036] | 1, 2, 3 |
| touch_3 | 50 | 0.082 | 0.054 | 0.095 | 1.15 | [0.004, 0.021] | 1, 2, 3 |

Also recorded (not tests). "Young by two" is the share of fawn and middle deer
catches made by two or more touchers (under the touch rule the only way to share a
young deer); section 1's runs are given for comparison, where the same column is the
share of young catches shared with a hunter near (the party rule).

| Run | Checkpoint | Coexisting | Adult ÷ chance | Adult − chance, 95% interval | Episodes adult > chance (of 20) | Adult energy share | Adults caught (50 episodes) | Hunter births (50 episodes) | Young by two |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| touch_1 | 250 | 50 | 1.16 | [0.015, 0.036] | 17 | 0.074 | 1277 | 1210 | 0.031 |
| | 500 | 50 | 1.10 | [0.006, 0.024] | 15 | 0.074 | 1283 | 1089 | 0.040 |
| | 1000 | 50 | 1.12 | [0.006, 0.026] | 16 | 0.050 | 721 | 891 | 0.029 |
| touch_2 | 250 | 50 | 1.20 | [0.017, 0.038] | 17 | 0.076 | 1293 | 1185 | 0.030 |
| | 500 | 50 | 1.14 | [0.012, 0.032] | 16 | 0.083 | 1421 | 1117 | 0.039 |
| | 1000 | 50 | 1.21 | [0.023, 0.036] | 19 | 0.070 | 1060 | 949 | 0.033 |
| touch_3 | 250 | 50 | 1.10 | [0.002, 0.020] | 17 | 0.052 | 811 | 1073 | 0.022 |
| | 500 | 50 | 1.07 | [−0.000, 0.015] | 13 | 0.048 | 697 | 920 | 0.022 |
| | 1000 | 50 | 1.15 | [0.004, 0.021] | 12 | 0.032 | 399 | 743 | 0.018 |
| run_1 (party) | 1000 | 43 | 1.20 | [0.002, 0.028] | 11 | 0.030 | 243 | 439 | 0.084 |
| run_2 (party) | 1000 | 49 | 1.23 | [0.024, 0.045] | 18 | 0.075 | 1130 | 920 | 0.164 |
| run_3 (party) | 1000 | 50 | 1.15 | [0.010, 0.027] | 16 | 0.055 | 706 | 740 | 0.129 |

**What section 2 shows.** Section 1's result held when only touching counts: 50 of 50
episodes coexisting in all three runs, at every checkpoint, and adults approached with
a partner near 1.12-1.21 times chance at iteration 1000, intervals above zero. So in
this world the party rule for young deer was not needed for coexistence or for adult
approaches above chance. Coexistence was, if anything, steadier than under the party
rule (150 of 150 against 142 of 150), though three runs per rule cannot establish a
difference.

**Limits of section 2.**

- **The effect is small**, as in section 1: adults are 1.5-3.2% of the deer caught at
  iteration 1000 and give 3-7% of the hunters' energy from deer.
- **Late decline again.** From iteration 500 to 1000, adult catches fell by 44% in
  touch_1, 25% in touch_2 and 43% in touch_3, hunter births with them, though no run
  lost its hunters. Section 1 saw the same in two of three runs; why is still not known.
- **touch_3 is borderline in places:** at iteration 500 its adult − chance interval
  touched zero, and at iteration 1000 adults beat chance in only 12 of 20 episodes. Its
  hunters also spent much less time near each other (chance 0.08 against 0.13-0.14).
- The measure is still approach with a partner near (section 1.1, limits), although
  under this rule a hunter near gets nothing. Teaming up on young deer is rare: 2-3%
  of young catches are made by two touchers.

**Section 2.1 verdict: pattern shown.** `launchers/energy_verdict.py`: touch_2 and touch_3
meet A, B and C; touch_1 meets A and C and misses B narrowly (interval [−0.001, +0.025]).

| Run | A: near (4,5) − near (2,3.5] | B: near (4.5,5) − chance | C: joint (4,5) ÷ joint adult | Tests met |
| --- | --- | --- | --- | --- |
| touch_1 | +0.043 [+0.029, +0.056] | +0.012 [−0.001, +0.025] | 0.92 [0.84, 1.02] | A, C |
| touch_2 | +0.059 [+0.048, +0.070] | +0.019 [+0.008, +0.029] | 0.86 [0.76, 0.97] | A, B, C |
| touch_3 (not looked at before) | +0.032 [+0.022, +0.043] | +0.013 [+0.002, +0.024] | 0.86 [0.71, 1.05] | A, B, C |
| *comparison, no test:* | | | | |
| run_1 (party) | +0.019 [+0.010, +0.029] | −0.007 [−0.014, +0.001] | 0.29 [0.22, 0.37] | A |
| run_2 (party) | +0.043 [+0.032, +0.055] | −0.027 [−0.036, −0.019] | 0.24 [0.20, 0.28] | A |
| run_3 (party) | +0.035 [+0.027, +0.044] | −0.023 [−0.033, −0.013] | 0.23 [0.18, 0.28] | A |
| touch_1, iteration 500 | +0.069 [+0.055, +0.081] | +0.028 [+0.017, +0.039] | 1.00 [0.91, 1.11] | A, B, C |
| touch_2, iteration 500 | +0.064 [+0.055, +0.072] | +0.022 [+0.011, +0.032] | 0.89 [0.82, 0.97] | A, B, C |
| touch_3, iteration 500 | +0.038 [+0.027, +0.049] | +0.004 [−0.005, +0.013] | 0.84 [0.76, 0.93] | A, C |

Near share ÷ chance and joint share by the deer's energy, final checkpoints (40
episodes each; from `results/<run>_energy.json`):

| Deer energy | touch_1 near ÷ chance | touch_2 | touch_3 | run_1 (party) | run_2 (party) | run_3 (party) | touch_1 joint | touch_2 | touch_3 | run_1 (party) | run_2 (party) | run_3 (party) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ≤ 2 (fawn) | 0.75 | 0.70 | 0.69 | 0.80 | 0.76 | 0.79 | 0.010 | 0.010 | 0.005 | 0.006 | 0.014 | 0.010 |
| 2-2.5 | 0.58 | 0.56 | 0.72 | 0.64 | 0.44 | 0.56 | 0.009 | 0.004 | 0.007 | 0.004 | 0.011 | 0.007 |
| 2.5-3 | 0.66 | 0.47 | 0.63 | 0.54 | 0.49 | 0.36 | 0.024 | 0.015 | 0.013 | 0.003 | 0.009 | 0.000 |
| 3-3.5 | 0.74 | 0.65 | 0.62 | 0.45 | 0.49 | 0.49 | 0.035 | 0.031 | 0.019 | 0.003 | 0.010 | 0.003 |
| 3.5-4 | 0.90 | 0.79 | 0.83 | 0.71 | 0.76 | 0.76 | 0.049 | 0.049 | 0.027 | 0.007 | 0.016 | 0.013 |
| 4-4.5 | 0.97 | 0.88 | 0.90 | 0.65 | 0.72 | 0.70 | 0.063 | 0.070 | 0.034 | 0.009 | 0.023 | 0.014 |
| 4.5-5 | 1.09 | 1.13 | 1.15 | 0.91 | 0.82 | 0.81 | 0.082 | 0.093 | 0.045 | 0.016 | 0.030 | 0.021 |
| ≥ 5 (adult) | 1.09 | 1.21 | 1.10 | 1.17 | 1.23 | 1.14 | 0.079 | 0.095 | 0.045 | 0.043 | 0.109 | 0.077 |

**What section 2.1 shows.** Under the touch rule, where nothing forces a partner for a
middle deer, hunters approached the largest middle deer (energy 4.5-5) with a partner
near slightly more often than chance (1.09-1.15 times) and teamed up on deer in (4, 5)
nearly as often as on adults (ratio 0.86-0.92). Under the party rule neither held:
the largest middle deer stayed below chance and joint touches on them were a quarter
of the adult level. touch_3, the one run not looked at before the tests were set,
met all three tests. Within this world, the pattern is consistent with hunters
pairing up where a partner pays without being required, the stag hunt in its strict
sense.

**Limits of section 2.1.**

- **Not an independent confirmation.** The tests were set after a first look at
  touch_1 and touch_2 (earlier checkpoints), with B and C's form and C's threshold
  informed by that look. touch_3 alone was not looked at; it met all tests, a single
  run.
- **Small effects.** B's margins are 1-2 percentage points; touch_1 missed B by 0.1
  points at its lower bound, and touch_3 missed B at iteration 500. Joint touches are
  rare: at most 9.5% of approaches in the touch-rule runs, adults included.
- **Not proof of choosing by payoff.** The rise crosses chance near energy 4.5, not at
  the payoff break-even 3.5. Large middle deer may share places with adults, or
  hunters may treat them like adults (they differ by energy only), which a C ratio
  near 1 would also show. A control world in which adults can be caught alone would
  separate place and generalisation from cooperation; it was not run (a control of
  this kind, with young deer caught alone for certain instead, is section 5).
- **The comparison with the party rule is between worlds trained apart,** so the
  differences in B and C may come from any difference the rule makes, not from payoff
  alone.

**Hard stop reached.** No world changes followed these runs; the next step is the
user's decision.

## 3. Logistic grass (pre-registered 2026-11-09, before training)

**Question:** do section 2's results hold when grass regrows gradually? In sections
1 and 2 an eaten patch was gone for 400 physics steps and then back whole at once.
Here it regrows logistically, as grass does. Two questions, judged separately:
does section 2's replication hold (coexistence and adult approaches with a partner
near above chance), and does section 2.1's pattern (partnering where a partner pays
but is not required) hold?

**World:** section 2's world (`deer_catch_rule: "touch"`) with `grass_regrowth:
"logistic"` (README.md, "Grass regrowth"; the default since commit 838357eb), growth
rate 0.0187, root reserve 0.01; nothing else changed (`launchers/check_logistic_world.py`
compares the settings with touch_1's, before training and for each saved run). An eaten patch regrows from bare
to 95% in about 400 physics steps, a deer eating a patch gains its biomass times 2,
and deer and hunters see the biomass of the nearest patch (one input more each). So
the deer's food supply differs from section 2's: grazed patches regrow slowly, and a
deer eating a patch early gets less. Deer energies, and with them the mix of fawns,
middle deer and adults, may change.

**Runs:** three runs from scratch (`logistic_1`, `logistic_2`, `logistic_3`), 500
iterations, 14 env runners, two at a time (`launchers/run_logistic.sh`, which calls
`launchers/replicate.py NAME 500 --catch-rule touch` and checks the saved world).

**Iterations: 500, not 1000, chosen after seeing sections 1 and 2.** There,
adult ÷ chance was above 1 from iteration 250 on and stayed at a similar level, while
adult catches and the ecology changed: they fell from iteration 500 to 1000 in all six
runs (by 13 to 84%). At iteration 500 each overall verdict would have been the same as
at 1000, though not each run's outcome (section 1: 3 of 3 runs met all tests instead of
2 of 3; section 2: 3 of 3 as at 1000; section 2.1: 2 of 3 as at 1000, with touch_3
instead of touch_1 missing B). A verdict at 500 therefore
describes the state before any such late decline; it says nothing about stability over
longer training. The final checkpoint is iteration 500; iteration 250 is recorded, not
tested.

**Tests and verdicts,** on the final checkpoints (iteration 500):

1. **Replication (section 1's tests, unchanged):** coexistence in at least 45 of 50
   episodes; adult share with a partner near above fawn share and above chance
   (coop_fair.py, 20 episodes); adult share ÷ chance above 0.90. Replicated if all three
   are met in at least 2 of the 3 runs (`launchers/verdict.py logistic_1 logistic_2
   logistic_3`).
2. **Partnering by energy (section 2.1's tests, unchanged):** A, B and C with their
   thresholds, 40 episodes (seeds 0-39), 5000 bootstrap resamples. Pattern shown if
   all three are met in at least 2 of the 3 runs (`launchers/energy_verdict.py`).

Both verdicts are reported, each on its own. Section 2.1's tests were set on touch-rule
runs; these three runs were not looked at when they were set, so a positive verdict 2
would be a test on fresh runs, though in a changed world (gradual grass), not an exact
repeat of section 2.1.

**Comparison, no test:** touch_1-3 at iteration 500 (section 2.2), same measures.

**Also recorded, not tests:** iteration 250; adults caught, hunter and deer births,
grass eaten, the share of young catches made by two touchers, the adult energy share.

**Hard stop:** no setting changes during or after these runs, whatever the outcome. In
particular, if coexistence fails, the grass growth rate is not tuned and the runs are
not repeated with other settings; that would be the result. What follows is the
user's decision.

**What outcomes would mean:** a replication would show section 2's result does not
need grass coming back whole at once, for this growth rate and root reserve; other
rates were not tried. A failure of coexistence would show this growth rate changes the
ecology enough to break it, in this world; it would not say gradual grass as such
prevents coexistence. A failure of verdict 2 with verdict 1 met could mean section
2.1's pattern is specific to instant grass or too small to recur, but also that gradual
grass changed the deer's energies and with them how many approaches fall in the tested
energy bands (test C divides by the adults' joint share), lowering the tests' power;
the bin counts are reported to judge this.

**Running:** `systemd-run --user --unit deer-logistic bash
/home/doesburg/Projects/PredPreyGrass/predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/launchers/run_logistic.sh`
(systemd-run needs the absolute path). Status in `runs/logistic.status`; results
to `results/logistic_*` and `results/logistic_verdict.txt`,
`results/logistic_energy_verdict.txt`.

### 3.1 Stopped early (2026-11-09, 11:03)

**No verdict.** The runs were stopped by the user's decision at iteration 287
(logistic_1) and 280 (logistic_2); logistic_3 never started. Section 3's tests were
not applied.

**Why.** Training showed hunters barely breeding: 0.06-0.07 hunter births per episode
at iteration 240-250 (touch_1 at the same point: 9.4), deer births 47-48 (touch_1:
313), and short training episodes (19 per iteration against 2-3), with no improvement
since about iteration 100. A look at logistic_1's iteration-240 checkpoint (4 episodes,
taken at 10:55, before stopping; it changed no test) found hunters extinct in all 4,
after 1825 to 3365 of 8000 physics steps, with about 23 deer on average (touch_3's
final checkpoint: about 57).

**What seems to cause it (exploratory, 3 episodes each).** Deer eat a patch by
touching it, whatever its biomass, so under logistic grass they grazed patches at a
mean biomass of 0.12-0.15, when regrowth from bare is still slow (0.13 after 147
physics steps with root reserve 0.01). Deer gained 0.21-0.25 grass energy per physics
step against 0.48 with instant grass (touch_1's final checkpoint), so section 3 changed
the food supply as well as the way grass regrows, and probably mostly tested the
former. The comparison is between different trained policies, so the numbers are only
indicative.

**What follows:** a calibration of logistic grass to section 2's food supply, decided
with the user, set before any new training run.

### 3.2 Calibrating logistic grass (pre-registered 2026-11-09, before any measurement)

**Agreed with the user:** one knob, one calibration, one pilot, then a hard stop.

**Knob:** the root reserve only (`grass_root_reserve`, b0), which sets how fast bare
grass starts regrowing; growth rate 0.0187, energy per patch and everything else stay.
A larger b0 means grass recovers from roots and stubble rather than from almost
nothing, closer to how grazed grass regrows.

**Target:** the deer's grass supply of section 2's world (instant grass), measured
with fixed behaviour: touch_1's final policies for hunters and deer, in the touch-rule
world, seeds 0-9, episodes of up to 8000 physics steps (`launchers/grass_supply.py`;
under logistic grass the extra biomass input is removed so the policies see what they
were trained with). Measure: grass energy gained by deer per physics step. Candidates:
b0 = 0.01, 0.02, 0.05, 0.1, 0.2, 0.5. **Rule:** the candidate whose supply ÷ instant
supply is closest to 1 (ties to the smaller) is chosen, whatever the other numbers.
Limit: behaviour is that of policies trained with instant grass; trained logistic-grass
deer may graze differently.

**Pilot:** one training run (`logistic_pilot`), touch rule, logistic grass with the
chosen b0, 150 iterations, 14 env runners. **Passes if** hunter births per episode
(training log, `births_predator`), averaged over iterations 141-150, are at least 2.57,
half the mean of touch_1-3 over the same iterations (5.39, 5.03, 4.99; mean 5.14).

**If it passes:** section 3's replication as pre-registered there (three runs of 500
iterations, its two verdicts and comparison), with only b0 changed; the runs are named
`logistic_b_1` to `logistic_b_3`, and the pilot does not enter them.
**If it fails:** the logistic-grass line ends; the default stays instant. No second
calibration, no other knob.

**Calibration result (2026-11-09, 11:13; `results/grass_supply.json`).** Grass energy
gained by deer per physics step, touch_1's final policies, 10 episodes each:

| Grass | Energy per step | ÷ instant | Biomass per patch eaten | Mean deer | Mean hunters | Hunters extinct (of 10) |
| --- | --- | --- | --- | --- | --- | --- |
| instant | 0.475 | 1.00 | 1.03 | 42.0 | 8.9 | 0 |
| b0 0.01 | 0.166 | 0.35 | 0.08 | 16.9 | 2.9 | 9 |
| b0 0.02 | 0.180 | 0.38 | 0.07 | 19.2 | 2.5 | 6 |
| b0 0.05 | 0.251 | 0.53 | 0.07 | 25.4 | 3.4 | 5 |
| b0 0.1 | 0.349 | 0.73 | 0.08 | 31.1 | 5.7 | 0 |
| b0 0.2 | 0.400 | 0.84 | 0.13 | 25.9 | 8.9 | 0 |
| b0 0.5 | 0.495 | 1.04 | 0.19 | 32.3 | 11.2 | 0 |

By the rule, **b0 = 0.5** (the only candidate near instant's supply). With it, bare grass
reaches 0.95 in 144 physics steps (400 with b0 0.01), so regrowth is gradual but about
three times faster than instant grass's 400-step wait, and deer keep patches grazed
short (0.19 of a patch per bite), more like a lawn than a meadow. "Instant" counts a
patch as 1 even when a deer ate two at once, hence biomass above 1. Pilot launched
11:13 (`logistic_pilot`, systemd unit deer-logistic-pilot).

**Amendment (2026-11-09, 11:20, user's decision, before the pilot's result):** if the
pilot fails, this approach (logistic grass tuned by root reserve) ends as set, but the
search for a more realistic grass than instant regrowth goes on as a new design (for
example partial grazing: deer crop a patch and leave stubble that regrows), with its own
pre-registration, pilot and hard stop. The default stays instant meanwhile.

**Pilot result (2026-11-09, 12:00): passed.** logistic_pilot (b0 0.5, 150 iterations):
hunter births per episode averaged over iterations 141-150 were 11.01 (bar 2.57;
touch_1-3 over the same iterations 4.99-5.39), deer births 203.1 (touch_1-3: 190-194).
So with the deer's grass supply restored, logistic grass at this root reserve supports
breeding hunters early in training, at about twice section 2's rate.

**Replication not run (user's decision, 2026-11-09, before the pilot's result):** b0 0.5
is a crude stand-in for grass regrowing from what grazers leave; the user chose to go
for the biologically more realistic mechanism instead (deer crop patches down to
stubble, section 4) rather than replicate with b0 0.5. Section 3 thus ends without a
verdict; its pilot shows only that gradual regrowth with section 2's food supply does
not by itself starve the hunters.

## 4. Grazed grass (pre-registered 2026-11-09, before any measurement)

**Question:** do section 2's results hold when deer graze as grazers do? Decided with
the user, who asked for the biologically most realistic mechanism after section 3.

**World:** section 2's world (touch rule) with `grass_regrowth: "grazed"` (grass.py):

- a patch's biomass grows logistically (growth rate 0.0187, root reserve 0.01);
- each physics step, a patch in reach is bitten by its closest deer, which takes up to
  `grass_bite` 0.1 of a full patch (a full patch is grazed in about one decision), but
  not below `grass_stubble`; it gains that share of 2 energy;
- a grazed patch keeps its place and regrows from its stubble;
- a patch kept below half grown for `grass_dieback_delay` 400 physics steps dies and
  re-seeds next to a half-grown patch of its cluster, from bare (sustained grazing moves
  grass, as eaten patches did under instant grass; light grazing does not);
- a cluster with no patch at half grown collapses and regrows elsewhere after 400
  physics steps, as before.

Bite 0.1 and dieback delay 400 are set, not tuned; the stubble height is calibrated.

**Calibration (as section 3.2):** `launchers/grass_supply.py 10 24 OUT stubble`,
touch_1's final policies, seeds 0-9; candidates stubble 0.05, 0.1, 0.2, 0.3, 0.4
(0.5 or more would make overgrazing impossible); the candidate whose supply ÷ instant
supply is closest to 1 is chosen (ties to the smaller). If even that is outside 0.8 to
1.25, the design fails at calibration (hard stop below).

**Pilot:** one run (`grazed_pilot`), 150 iterations, passes if hunter births per
episode over iterations 141-150 average at least 2.57 (section 3.2's bar).

**Replication, if the pilot passes:** three runs `grazed_1` to `grazed_3`, 500
iterations, with section 3's tests, verdicts and comparison unchanged (section 1's
three tests, replicated in at least 2 of 3; section 2.1's A, B and C, pattern in at
least 2 of 3; touch_1-3 at iteration 500 for comparison, no test). The pilot does not
enter them.

**Hard stop:** if calibration or the pilot fails, the grazed design ends with no
tuning of bite, dieback or anything else; the default stays instant. If the
replication runs, no setting changes whatever the outcome. What follows is the user's
decision.

### 4.1 Calibration (2026-11-09, 12:05): failed by the rule

`results/grass_supply_stubble.json`, touch_1's final policies, 10 episodes each:

| Grass | Energy per step | ÷ instant | Biomass per bite | Mean deer | Mean hunters | Hunters extinct (of 10) | Collapses | Diebacks |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| instant | 0.475 | 1.00 | 1.03 | 42.0 | 8.9 | 0 | 380 | 0 |
| stubble 0.05 | 0.192 | 0.40 | 0.03 | 20.1 | 3.1 | 8 | 96 | 569 |
| stubble 0.1 | 0.214 | 0.45 | 0.03 | 25.1 | 2.7 | 3 | 116 | 713 |
| stubble 0.2 | 0.183 | 0.39 | 0.04 | 22.4 | 2.8 | 8 | 150 | 156 |
| stubble 0.3 | 0.160 | 0.34 | 0.05 | 20.0 | 2.8 | 8 | 155 | 7 |
| stubble 0.4 | 0.183 | 0.38 | 0.05 | 21.9 | 3.2 | 8 | 150 | 0 |

The closest candidate, stubble 0.1, gives 0.45 of instant grass's supply, outside 0.8 to
1.25: by section 4's rule the design fails at calibration.

**A likely flaw in the measure (stated after the result).** The supply hardly changes
with the stubble height, and bites are tiny (0.03-0.05 of a patch). touch_1's deer were
trained on instant grass, where a touch gives a whole patch at once; under grazed grass
a whole patch takes about 8 physics steps of staying, which these deer never learned.
So the fixed-behaviour measure probably shows the old deer's habits, not what grazed
grass can supply. Under section 3.2's logistic grass this did not arise, as deer still
ate a whole patch on touch. This was not foreseen when section 4 was pre-registered.
How to proceed is the user's decision.

**Amendment (2026-11-09, 12:10, user's decision, after seeing the calibration
result):** the fixed-behaviour calibration is set aside for the grazed design, as it
likely measures instant-trained deer's habits. The pilot goes ahead with stubble 0.1
(the rule's choice) and trains deer and hunters from scratch, so it measures supply
with deer that can learn to graze. Pilot and its bar unchanged (2.57). This is the only
amendment: if the pilot fails, the grazed design ends.

### 4.2 Pilot (2026-11-09, 13:06): failed; the grazed design ends

grazed_pilot (stubble 0.1, 150 iterations): hunter births per episode over iterations
141-150 averaged 0.98 (bar 2.57; touch_1-3 4.99-5.39; section 3.2's pilot 11.01), deer
births 57.6 (touch_1-3: 190-194). Deer trained from scratch on grazed grass stayed few
and the hunters barely bred, so the shortage was not only the fixed-behaviour
calibration's. By the hard stop, the grazed design ends; no setting is tuned.

### Gradual grass: conclusion (2026-11-09)

Instant grass stays the default, as decided with the user; sections 1 and 2's results
stand on it. Of three gradual designs, two (logistic grass with root reserve 0.01,
grazed grass) gave the deer roughly 40% of instant grass's food and starved the hunters;
the one that fed them (root reserve 0.5) regrows so fast that it is little more
realistic than instant grass, and was not replicated. In this world the energy budget
is tuned to whole patches returning at once: realistic grazing would need a re-balanced
ecosystem (more grass, more energy per bite, or other changes), a project of its own,
not a different growth rule. `"logistic"` and `"grazed"` stay in the code, untested for
cooperation.

## 5. Control world: young deer caught alone for certain (pre-registered 2026-11-09, before any code or training)

**Question:** section 2.1 found hunters approaching the largest middle deer (energy
4.5-5) with a partner near above chance, and teaming up on deer in (4, 5) nearly as
often as on adults. It could not tell apart three explanations (section 2.1, limits):

1. **payoff:** a partner pays for a middle deer above energy 3.5 (the strict stag hunt);
2. **place:** big middle deer are where adults are, and hunters partner there anyway;
3. **generalisation:** hunters treat big middle deer like adults, which differ by
   energy only.

**World:** section 2's touch world (instant grass) with one change: a lone touch catches
a fawn or a middle deer for certain (`deer_young_solo: "certain"`; section 2: chance
falling from 1 at energy 2 to 0 at energy 5). Adults are unchanged: caught only by two
touchers within 8 physics steps. Two touchers still catch any deer together and share
it. So a partner never pays for a young deer (a shared meal is half of a certain lone
one), while places and adults, and so explanations 2 and 3, stay as they were.

**Prediction:** if explanation 1 carries section 2.1's pattern, it disappears here. If
explanations 2 or 3 carry it, it stays.

**Runs:** three runs from scratch (`control_1`, `control_2`, `control_3`), 500 iterations
(as agreed for section 3), 14 env runners, two at a time, then the third.

**Validity:** young deer are easier to catch here, so the hunters get more food and may
eat the deer out. A run counts only if its final checkpoint coexists in at least 45 of
50 episodes (`evaluate.py`). Fewer than 2 valid runs: the control is inconclusive and
stops (hard stop below).

**Tests,** on each valid run's final checkpoint, with section 2.1's measure and tests A,
B and C unchanged (`launchers/coop_by_energy.py`, 40 episodes, seeds 0-39, 95% bootstrap
intervals over episodes).

**Verdict** (`launchers/energy_verdict.py`, plus the two rules below):

- **Payoff supported:** in at least 2 valid runs, B is not met **and** C's interval lies
  entirely below 0.5 (teaming on big middle deer well under the adult level, as under
  the party rule, 0.23-0.29). Section 2.1's pattern then needs the payoff of a partner
  for middle deer; place and generalisation alone do not produce it.
- **Payoff not supported:** in at least 2 valid runs, A, B and C are all met. Section
  2.1's pattern then appears without that payoff, and its stag hunt reading is
  withdrawn.
- **Inconclusive:** anything else.

**Comparison, no test:** touch_1-3 at iteration 500 (the same training length):
touch_1 and touch_2 met A, B and C there, touch_3 met A and C. Also recorded: section
1's tests 1-3 (adults with a partner near above chance should still hold, as adults
still need two), joint share by energy bin, adults caught, hunter births, and young
catches made by two.

**Limits, set now:**

- The worlds differ in food as well as payoff: certain catches feed the hunters more,
  which may change their numbers, spacing and so chance. Chance is measured per world,
  but a difference in B or C may still come partly from ecology.
- Three runs per world; a "payoff supported" verdict rests on the absence of a pattern
  in this world, shown by C's upper bound, not by a direct test of the difference
  between worlds.
- Place and generalisation are not separated from each other; only both from payoff.

**Running:** `launchers/run_control.sh` (systemd unit `deer-control`, status
`runs/control.status`) checks the settings against touch_1's before and after training,
evaluates iterations 250 and 500, and writes `results/control_verdict.txt` (section 5's
verdict, valid runs only) and, recorded only, `results/control_section1_tests.txt` and
`results/control_energy_tests.txt` (all three runs).

**Hard stop:** no setting changes during or after these runs, whatever the outcome,
including if the validity rule fails. What follows is the user's decision.

### 5.1 Results (2026-11-10, 01:50): payoff supported, in all 3 runs

All three runs are valid (coexisting 48, 49 and 49 of 50 episodes), and in all three, B
is not met and C's interval lies below 0.5 (`results/control_verdict.txt`).

| Run | Coexisting (of 50) | A: near (4,5) − near (2,3.5] | B: near (4.5,5) − chance | C: joint (4,5) ÷ joint adult | Pattern absent |
| --- | --- | --- | --- | --- | --- |
| control_1 | 48 | +0.117 [+0.102, +0.133] | −0.013 [−0.026, +0.001] | 0.31 [0.27, 0.35] | yes |
| control_2 | 49 | +0.096 [+0.083, +0.110] | −0.021 [−0.036, −0.007] | 0.28 [0.24, 0.32] | yes |
| control_3 | 49 | +0.116 [+0.102, +0.129] | −0.028 [−0.043, −0.013] | 0.28 [0.23, 0.32] | yes |
| *comparison, no test:* | | | | | |
| touch_1, iteration 500 | 50 | +0.069 [+0.055, +0.081] | +0.028 [+0.017, +0.039] | 1.00 [0.91, 1.11] | |
| touch_2, iteration 500 | 50 | +0.064 [+0.055, +0.072] | +0.022 [+0.011, +0.032] | 0.89 [0.82, 0.97] | |
| touch_3, iteration 500 | 50 | +0.038 [+0.027, +0.049] | +0.004 [−0.005, +0.013] | 0.84 [0.76, 0.93] | |

Near share ÷ chance and joint share by the deer's energy, final checkpoints (40
episodes each; from `results/<run>_energy.json`; touch runs at iteration 500):

| Deer energy | control_1 near ÷ chance | control_2 | control_3 | touch_1 | touch_2 | touch_3 | control_1 joint | control_2 | control_3 | touch_1 | touch_2 | touch_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ≤ 2 (fawn) | 0.79 | 0.80 | 0.86 | 0.71 | 0.68 | 0.71 | 0.028 | 0.024 | 0.027 | 0.013 | 0.011 | 0.007 |
| 2-2.5 | 0.23 | 0.29 | 0.31 | 0.56 | 0.46 | 0.43 | 0.007 | 0.007 | 0.005 | 0.033 | 0.023 | 0.000 |
| 2.5-3 | 0.31 | 0.37 | 0.23 | 0.62 | 0.68 | 0.57 | 0.000 | 0.003 | 0.003 | 0.027 | 0.023 | 0.014 |
| 3-3.5 | 0.45 | 0.44 | 0.29 | 0.69 | 0.69 | 0.65 | 0.005 | 0.003 | 0.003 | 0.035 | 0.044 | 0.023 |
| 3.5-4 | 0.71 | 0.76 | 0.70 | 0.87 | 0.91 | 0.82 | 0.014 | 0.009 | 0.012 | 0.057 | 0.062 | 0.035 |
| 4-4.5 | 0.79 | 0.84 | 0.85 | 1.01 | 0.99 | 0.86 | 0.043 | 0.037 | 0.039 | 0.088 | 0.078 | 0.043 |
| 4.5-5 | 0.94 | 0.90 | 0.86 | 1.18 | 1.13 | 1.04 | 0.061 | 0.041 | 0.041 | 0.112 | 0.105 | 0.058 |
| ≥ 5 (adult) | 1.13 | 1.09 | 1.20 | 1.09 | 1.14 | 1.07 | 0.170 | 0.142 | 0.146 | 0.100 | 0.103 | 0.061 |

Also recorded (not tests): section 1's tests 1-3 met in all three runs (adults ÷ chance
1.12, 1.10, 1.17, intervals above zero); chance 0.20-0.23 (touch runs at iteration 500:
0.11-0.16); adults caught in 50 episodes 1133, 992, 947 (touch: 697-1421); hunter births
1098-1125 (touch: 920-1117); young deer caught by two 1.2-1.5% of young catches (touch:
2.2-4.0%).

**What section 5 shows.** Where a partner never pays for a young deer, hunters did not
approach the largest middle deer with a partner near above chance (0.86-0.94 times
chance, against 1.04-1.18 in the touch world at the same training length), and teamed
up on deer in (4, 5) at under a third of the adult level (against 0.84-1.00). So
place and likeness to adults, which this world keeps, did not by themselves produce
section 2.1's pattern; the pattern is consistent with hunters partnering more where a
partner pays.

**Limits of section 5.**

- **The rise is there without the payoff.** Test A, the rise from small to large
  middle deer, was met here too, more strongly than in the touch world (+0.10 to +0.12):
  near share rises with energy even where a partner never pays, which is consistent
  with place or likeness producing part of section 2.1's rise. What this world lacks is
  the step from just below chance to above it; that the payoff causes the step is
  consistent with these results but not shown by them (next limits).
- **C's drop is partly in the denominator.** Joint touches on adults were more common
  here (0.14-0.17 against 0.06-0.10), and those on deer in (4, 5) less common (about
  0.04-0.06 against 0.04-0.11; touch_3's overlaps the control range, below control_1's
  and above control_2's and control_3's).
  C's ratio falls from both sides, so it overstates the change in teaming up on big
  middle deer alone. B, measured against chance in its own world, does not have this
  problem; it is the cleaner of the two.
- **The worlds differ in ecology.** Hunters were closer together here (chance 0.20-0.23
  against 0.11-0.16), and fewer young deer were caught; any of that may affect B and C
  as well as the payoff.
- **Small margins**, as in section 2.1: B differs between worlds by 2-6 percentage
  points over all pairings of a control run with a touch run (3-4 for runs of the same
  number).
- **Absence, not a test of difference.** As set: the verdict rests on the pattern's
  absence in this world, not on a direct test of the difference between the worlds;
  touch_3 at iteration 500 also missed B.
- Place and likeness are not separated from each other.

**Hard stop reached.** No setting changes followed; the next step is the user's
decision.

## Closed (2026-11-10)

The module is closed by the user's decision, with positive results, and stays in
place (not archived). What it shows, with the hedges of each section:

- **Section 1:** the replication of stag_hunt_rule_prey's cooperation result without
  calls: met in 2 of 3 runs at iteration 1000 (all 3 at 250 and 500).
- **Section 2:** with the touch rule, hunters and deer coexisted in 50 of 50 episodes in
  all 3 runs, and adult approaches with a partner near were above chance (1.12-1.21).
- **Section 2.1:** hunters partnered more on large middle deer, where a partner pays,
  than on small ones: pattern shown in 2 of 3 runs, small margins.
- **Section 5:** in a control world where a partner never pays for a young deer, that
  step was absent in 3 of 3 runs: consistent with the payoff driving it, though the
  rise with deer size persists there.
- **Sections 3-4:** gradual grass closed; instant grass stays the default.

Left unexplored: the late decline in adult catches between iterations 500 and 1000
(sections 1.1 and 2), whose cause is unknown. Continued in
eco_evolutionary_deer_stag_hunt (nature and nurture), which reads this module's
touch runs.
