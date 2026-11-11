# deer_food_sharing: results

## 1. Sharing big game (pre-registered 2026-11-11, before any training run)

Design and reasons: DESIGN.md.

### 1.1 World and runs

- **World:** deer_stag_hunt's touch world (its RESULTS.md section 2) with adult
  catches shared: s = 0.5 of the food to the other hunters within R = 128 of the deer
  (DESIGN.md "The rule"). Young deer are not shared.
- **Runs:** share_1, share_2, share_3, 1000 iterations each, deer_stag_hunt's PPO
  settings, fresh weights, episodes from the operating system's entropy. share_1 and
  share_2 side by side, then share_3 (launchers/run_sharing.sh).
- **Comparison:** deer_stag_hunt's touch_1-3 (the same world without sharing), from
  its committed results files.
- **Evaluation:** at iterations 250, 500 and the final checkpoint, evaluate.py 50
  episodes (seeds 0-49) and coop_fair.py 20 episodes, as for touch_1-3.

### 1.2 Measures

At the final checkpoint of each run:

- **coexisting:** episodes of 50 with both species alive at the last physics step;
- **C:** adult approaches with a partner near, over chance (coop_fair.py
  `adult_over_chance`); touch_1-3: 1.119, 1.206, 1.150;
- **A:** adults caught per episode; touch_1-3: 14.42, 21.20, 7.98.

### 1.3 Tests, set before any training run

- **Test 1 (condition):** every sharing run coexists in at least 45 of 50 episodes.
- **Test 2 (primary):** every sharing run's C is above every touch run's (so above
  1.206).
- **Test 3 (secondary):** every sharing run's A is above every touch run's (so above
  21.20).

"All three above all three" is complete separation, the strictest ordering three runs
a side allow. It is a pre-registered criterion, **not an exact permutation test**:
touch_1-3 are historical controls, trained earlier, and their values were known when
the criterion was chosen, so the six runs were not randomly assigned to the two worlds.
Had they been, complete separation would have a one-sided p of 1/20 = 0.05. The code
path is the same (this module without sharing reproduces touch_1's evaluation episode
for episode); the time, machine load and scheduling of the runs are not. Tests 2 and 3
are judged separately.

**Verdict wording:** tests 1 and 2 met: "in this world, every run with big game shared
among nearby hunters partnered on adults more than every one of the three earlier runs
without sharing, and coexisted"; test 3 met: "... and caught more adults". Not met:
"with sharing at s = 0.5 within R = 128, partnering on adults was not above every run
without sharing". The comparison is with historical controls, so a met test is
consistent with sharing increasing partnering, not a demonstration of it; no mechanism
(insurance, kin, free-riding) is claimed either way.

**Also reported, not tested:** S (adult attempt success); chance (hunter time with a
partner near, a measure of clustering); hunter births per episode; the share of adult
energy that went to bystanders (`adult_energy_to_bystanders`); adult catches with a
bystander and bystanders per such catch; the values at 250 and 500; C by deer energy if
run (coop_by_energy.py).

### 1.4 Failures that aren't results, and the hard stop

- A crash or hardware failure: the run is rerun from scratch once, the failed attempt
  kept as runs/<run>.crash1 and recorded in runs/sharing.crashes (run_sharing.sh does
  this); no other rerun. A code bug (not a setting): fixed, recorded, every affected run rerun.
- **Hard stop:** after the three runs, this stage stops whatever the outcome. No other
  value of s or R, no other sharing rule. If test 2 is not met, the sharing line ends.
  If it is met, an evolutionary follow-up is a new user decision.
