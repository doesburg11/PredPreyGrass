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
| run_1 | 250 | 50 | 1.25 | [0.035, 0.053] | 19 | 0.101 | | |
| | 500 | 50 | 1.20 | [0.020, 0.049] | 17 | 0.089 | 1481 | 1042 |
| | 1000 | 43 | 1.20 | [0.002, 0.028] | 11 | 0.030 | 243 | 439 |
| run_2 | 250 | 50 | 1.19 | [0.018, 0.032] | 20 | 0.064 | | |
| | 500 | 50 | 1.18 | [0.019, 0.036] | 18 | 0.079 | 1298 | 1046 |
| | 1000 | 49 | 1.23 | [0.024, 0.045] | 18 | 0.075 | 1130 | 920 |
| run_3 | 250 | 50 | 1.21 | [0.024, 0.047] | 18 | 0.082 | | |
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
- Three runs show run-to-run variation; they are not a population estimate.

**Hard stop reached.** No world changes followed; the next step is the user's decision.
