# predator_compound_provisioning — results

Follow-up module to [`predator_sexual_reproduction`](../predator_sexual_reproduction/RESULTS.md) (Iterations 0-19).
Numbering continues from there.

## Iteration 20 — Compound movement + gift-bit actions (pre-registered 2026-11-15, before any training)

**Why.** Iteration 19 made the mate gift a learned action. GIVE runs made real transfers and had more births than INERT
in all three seed pairs (H2 met), but the state-selective test (H1) failed: seed 44's male gap was +0.007 against a
+0.02 floor. Action 9 was also the only second way to stand still, and both arms chose it often (54-101 per 1000
predator-steps) mostly as a generic no-move action. That dilutes any state-selective signal. This iteration removes
that confound: the gift becomes a bit attached to every move, so giving never costs a move and can't be used as a
stand-still action.

**Code** (this module; it reuses `predator_sexual_reproduction`'s environment, gift resolution and PPO setup):

- `compound_gift_env.py`: predators have 18 actions. Action `a` makes move `a % 9`; `a >= 9` also requests a gift after
  movement. Prey keep 9 actions. `compound_gift="on"` transfers `learned_gift_amount` (2.0) energy to the recorded mate
  under the Iteration 19 rules (mate alive with energy > 0, within 3 cells after movement, giver keeps energy > 0;
  simultaneous resolution from one energy snapshot). `"inert"` has the same 18 actions and never transfers.
  `compound_env_creator` is the RLlib env creator. A first version patched the core tune script's env class, which reached
  only the driver process: a 1-iteration smoke run with a remote env runner crashed with `KeyError: 9` (workers had built
  the 9-action base env). The earlier smoke run had used no remote runners and missed it. Fixed, and re-smoked with 2
  remote runners (2054 steps sampled, no errors).
- `analyze_compound_gift.py`: per episode, realised gifts and gift-bit requests per 1000 predator-steps (also per sex),
  the gift-bit probability (sum of actions 9-17) in eligible states minus all other predator states, by sex, and
  Iteration 18's pair outcomes (births per 1000 predator-steps, post-birth proximity). An eligible state means the
  recorded mate is alive within 3 cells with energy in (0, 12), and the giver would keep energy > 0 after giving 2.
  Eligibility is read before the step, the transfer after movement. Intervals are episode-cluster bootstraps.
- `run_iteration20.sh`: the staged launcher (below). Tests: `tests/` (21) plus the 175 of `predator_sexual_reproduction`,
  196 in total, all passing.
- Review: the Codex second opinion could not run before launch (usage limit until 01:53). I reviewed the sentinel,
  the remote-runner env creation and the launcher myself before launch. That review fixed a launcher stop path that
  would have left training processes running after a kill. A Codex review of this committed code is queued for after
  01:53. If it finds an environment or training bug, the runs are stopped and this pre-registration is re-issued. If
  it finds an analysis-only bug, the fix is recorded here before any final result is read.

**Design.** Six runs: `COMPOUND_GIVE_K05_MB1024_SEED{42,43,44}` (`compound_gift=on`) and
`COMPOUND_INERT_K05_MB1024_SEED{42,43,44}` (`compound_gift=inert`). Both arms have the mate observation channel, the same
18-action predator policy, `male_gift_donation_rate=0` and parental care as shipped, so the only treatment difference is
whether a requested gift transfers. Otherwise this is Iteration 19's configuration: PROP k = 0.5 (flat forage rewards 0,
energy reward 0.5, combat-death penalty 0.2), default hunting odds, base environment, minibatch 1024, 30 epochs,
300 iterations. Three runs train at a time (GPU fraction 0.33, 4 env runners x 2 CPUs, 2 main CPUs). Wave 1 is GIVE 42,
INERT 42 and GIVE 43; wave 2 (INERT 43, GIVE 44, INERT 44) starts only after the halfway check passes and wave 1 ends.
Final analysis: checkpoint 29 (iteration 300), 20 episodes, evaluation seeds 7000-7019.

**Pre-registered tests.** Thresholds are carried over unchanged from Iteration 19.

- **H1, learned selective provisioning.** For the same sex in all three seeds: the GIVE gift-bit gap (eligible minus
  other) is >= 0.02 with its 95% interval excluding zero, AND it exceeds the matched-seed INERT gap of that sex by >= 0.02.
  The realised gift rate is reported but not tested: an untrained 18-action policy requests a gift in about half its
  steps, so transfers happen without any learning (smoke checkpoint: about 5 per 1000 predator-steps).
- **H2, benefit.** Births per 1000 predator-steps are higher in GIVE than in INERT in all three matched seed pairs
  (complete paired separation; one-sided exact sign test p = 0.125 with three pairs. This is a stringent directional
  criterion, not conventional significance).
- **Verdict.** H1 + H2: learned selective mate provisioning with a reproductive benefit. H1 without H2: learned selective
  giving, no shown benefit. H2 without H1: arm difference not attributable to learned selective giving (as Iteration 19).
  Otherwise: not met. Which sex gives, gift rates, request rates, post-birth proximity, population size and pair
  outcomes are descriptive only.

**Halfway kill rule** (efficiency rule, not a result test). When checkpoint 14 (about iteration 150) exists for GIVE 42,
GIVE 43 and INERT 42, run the analysis on those three (10 episodes, seed 7000). Continue only if, in at least one of
GIVE 42/43, at least one sex has a gift-bit gap at least 0.02 above INERT 42's gap for that sex. Otherwise stop wave 1
and don't start wave 2. If it passes, nothing changes.

**Hard stop.** One round: these six runs and the analysis above. No change to rewards, actions, gift amount, odds,
seeds, duration, thresholds or analysis after seeing any result, including the halfway read. If the verdict is not met,
this line of mate-provisioning experiments closes.

**Caveats fixed in advance.** Three seeds give no power for small effects. The two arms differ in ecology from step one:
untrained GIVE policies already transfer energy at random, INERT never does. That is why H2 alone can't support
the verdict, and H1 requires the gap to exceed INERT's. The gap compares eligible states with all other states. It shows
that the policy's gift choice depends on whether a gift can help the mate, not why. Eligibility is read before movement
and the transfer after it, so some eligible-state requests transfer nothing and some other-state requests do.

**Results (2026-11-16): stopped by the halfway kill rule. Verdict: not met. The mate-provisioning line closes.**

Wave 1 reached checkpoint 14 at 01:37. The analysis ran on 10 episodes (seeds 7000-7009) and the rule fired at 01:42.
All three runs were stopped, and wave 2 (INERT 43, GIVE 44, INERT 44) was never started. Source:
`runs/analysis/halfway.json`, `halfway.log`, `run_all.log`.

Gift-bit gap (eligible minus other states), with 95% episode-cluster bootstrap intervals:

| run | male gap | female gap | gift-bit requests / 1000 predator-steps | realised gifts / 1000 predator-steps | births / 1000 predator-steps |
| --- | --- | --- | --- | --- | --- |
| GIVE 42 | -0.0025 [-0.010, 0.006] | -0.0251 [-0.049, -0.004] | 502 | 20.4 | 3.16 [2.83, 3.56] |
| GIVE 43 | +0.0157 [0.008, 0.024] | -0.0210 [-0.038, -0.005] | 466 | 20.3 | 3.26 [3.00, 3.59] |
| INERT 42 | +0.0149 [0.006, 0.022] | -0.0158 [-0.041, 0.011] | 530 | 0 | 2.82 [2.63, 3.01] |

GIVE minus INERT 42 gap: male -0.017 (GIVE 42) and +0.001 (GIVE 43); female -0.009 and -0.005. The kill rule needed
at least one value >= 0.02, and the largest was +0.001.

Reading (descriptive; the pre-registered H1/H2 were never tested, because the runs stopped at the halfway rule):

- At iteration 150 the gift bit is still close to a coin flip: requests are 47-53% of predator steps in both arms.
  PPO has not made the bit depend on the mate's state in GIVE any more than in INERT, where the bit does nothing.
- The gaps that are there show up in INERT too. Females request slightly *less* in eligible states in all three runs, and
  the INERT male gap (+0.015) matches GIVE 43's. These are state-correlated patterns in an arbitrary action half, not
  provisioning learned because it pays.
- GIVE had more births than INERT 42 (3.16 and 3.26 vs 2.82), and females gave about twice as often as males (28-31 vs
  16-17 realised gifts per 1000 sex-steps). This matches what the Iteration 19 step-0 screen showed for untrained
  giving: random transfers alone raise births. It says nothing about learned selectivity, and it is the "H2 without H1"
  pattern that the pre-registration ruled out as evidence.
- Caveat: one INERT seed at the halfway checkpoint, 10 episodes. A late-emerging gap after iteration 150 can't be
  excluded. The rule was set in advance to trade that risk for compute.

**Line closed.** Under the hard stop, the mate-provisioning line (Iterations 17-20 across this module and
`predator_sexual_reproduction`) closes. In three rounds, from a fixed gift to a learned action to the compound gift
bit, a transfer to the mate raised births whenever it happened, and PPO never learned to target it to states where it
helps.
