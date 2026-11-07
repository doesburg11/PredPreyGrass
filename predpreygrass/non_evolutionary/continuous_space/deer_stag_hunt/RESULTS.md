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
