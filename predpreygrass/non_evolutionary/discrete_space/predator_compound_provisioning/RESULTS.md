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

**Codex review (queued before launch; ran 01:55, after the kill).** It found no environment or training bug. The remote
env creator, the `give_action_id=0` sentinel, movement decoding (`a % 9`), and arm parity were all checked and judged
sound. Analysis finding: "eligible" checks the giver's energy before the step. The environment charges homeostatic
and movement costs before it validates a gift, so a giver with just over 2 energy counts as eligible but can't give.
This affects only near-starving givers, a small share of states, and it is identical in GIVE and INERT. It can't
explain a GIVE-minus-INERT gap of at most +0.001, and it doesn't touch the main reading: the gift bit stayed close to
50% everywhere. It is recorded here and not re-analysed, under the hard stop. Its other points (per-sex request rates
and the full pair outcomes not reported, no automated verdict, launcher failure handling on the wave-2 and final paths)
don't affect the halfway read. The kill path did stop all training processes.

## Iteration 22 — INERT control to 300 iterations (pre-registered 2026-11-16, before any training)

**Why.** Iteration 21 ([`predator_mate_regard`](../predator_mate_regard/RESULTS.md)) trained fresh w = 0 runs,
which are this module's GIVE arm (compound gift on, no mate-regard reward), to 300 iterations. Their males developed a
gift-bit gap of +0.087, +0.060 and +0.140 (seeds 42-44) without any motivation. Iteration 20 asked whether GIVE beats
INERT, but its halfway kill rule stopped it before INERT reached 300 iterations. That rule rested on one reference run,
and Iteration 21 showed a run like that can be off by about 0.05. This iteration reopens that line once, at the user's
request, to train the missing control. The male focus and this test come from Iteration 21's result, so they are a
data-informed follow-up, not an independent prediction.

**Design.** Three runs `COMPOUND_INERT300_K05_MB1024_SEED{42,43,44}` (`compound_gift=inert`, otherwise Iteration 20/21's
configuration, 300 iterations, three at a time). Same code version as the Iteration 21 W0 runs (a1854b09; env and
analysis unchanged since). The comparison arm is Iteration 21's `MATEREGARD_W0_K05_MB1024_SEED{42,43,44}`, whose final
numbers already exist (`predator_mate_regard/runs/analysis/final.json`). Final analysis: checkpoint 29, 20 episodes,
seeds 7000-7019, the amended `analyze_compound_gift.py` (same settings as the W0 numbers). Launcher:
`run_iteration22.sh`, output `runs/analysis22/`.

**Pre-registered tests.**

- **H1, learned selective provisioning (males).** In all three seeds, the GIVE (w = 0) male gap exceeds the matched INERT
  male gap by >= 0.02. The GIVE male gaps already have 95% intervals excluding zero.
- **H2, births.** GIVE births per 1000 predator-steps exceed INERT's in all three matched pairs. GIVE values:
  3.45, 3.14, 4.40.
- **Verdict.** H1 + H2: males learn selective mate provisioning without a motivation, with a reproductive benefit.
  H1 alone: learned selective provisioning, no shown benefit. Otherwise: not met. Female gaps and gift rates are
  descriptive.

**Halfway kill rule.** At checkpoint 14, analyse the three INERT runs (10 episodes, seed 7000) and compare with the
fresh W0 runs at checkpoint 14 under the same settings (`exploratory_w0fresh_ckpt14.json`: male gaps +0.058, +0.031,
+0.060). Continue only if, in at least one seed, the W0 male gap exceeds the INERT male gap by >= 0.02. The reference
comes from the same code version, and this rule is deliberately lenient given about +/-0.05 run-to-run noise.

**Hard stop.** One round, no changes after seeing any result. If not met, the mate-provisioning line stays closed.

**Caveats.** The INERT runs train in a later batch than the GIVE runs, though with the same code and seeds; run-to-run
variance at equal seeds is about 0.05 on this gap, as Iteration 21 showed. Three seeds give no power for small
effects.

**Results (2026-11-17): H1 and H2 met their pre-registered rules.** In the registered wording, the verdict is
"males learn selective mate provisioning, with a reproductive benefit". What the design supports is narrower. Trained
GIVE males' gift requests are associated with pre-movement gift eligibility far more strongly than INERT males'
requests, at w = 0 (no added mate-regard reward). GIVE runs also have more births. This is a data-informed follow-up,
and the birth gain can't be attributed to the selectivity (caveats below).

All three INERT runs completed 300 iterations. The halfway check passed at 01:45 (W0-minus-INERT male gaps +0.069,
+0.009, +0.064); the final analysis finished at 03:16. Sources: `runs/analysis22/final.json`, `final.log`,
`halfway.json`, `run_all.log`. GIVE numbers: `predator_mate_regard/runs/analysis/final.json` (W0 arm).

Checkpoint 29, 20 episodes, 95% episode-cluster bootstrap intervals:

| seed | GIVE male gap | INERT male gap | male diff | GIVE female gap | INERT female gap | births GIVE | births INERT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42 | +0.087 [0.076, 0.097] | -0.020 [-0.025, -0.013] | +0.107 | -0.107 | +0.004 | 3.45 | 2.87 [2.72, 3.03] |
| 43 | +0.060 [0.054, 0.066] | +0.016 [0.008, 0.024] | +0.044 | -0.150 | +0.005 | 3.14 | 2.59 [2.34, 2.81] |
| 44 | +0.140 [0.129, 0.150] | -0.010 [-0.016, -0.003] | +0.149 | -0.359 | -0.038 | 4.40 | 2.88 [2.74, 3.02] |

- **H1 met.** The male gap exceeds INERT's by >= 0.02 in all three seeds (+0.044 to +0.149), and the GIVE intervals
  exclude zero. INERT male gaps stay near zero (-0.020 to +0.016). GIVE males request the gift bit more in eligible
  states than in other states; INERT males, with the identical but non-transferring action, do not. This is an
  observational contrast, not a manipulation of eligibility. GIVE also changes the ecology: energy, births,
  predator numbers, spatial arrangement and how often states are eligible differ between arms. The "other" category
  is a mixed set of states whose composition can differ between arms. So part of the cross-arm difference could
  come from different state distributions, not from a different response to eligibility.
- **H2 met.** GIVE births exceed INERT's in all three pairs (one-sided exact sign test p = 0.125 with three pairs; a
  stringent directional criterion, not conventional significance).

**Descriptive.**

- The estimated GIVE male gap was larger at checkpoint 29 than at checkpoint 14 in all three runs: +0.058, +0.031,
  +0.060 rose to +0.087, +0.060, +0.140. Checkpoint 14 used 10 episodes and checkpoint 29 used 20, and there is no
  uncertainty analysis of the change and no intermediate trajectory. INERT was near zero at both checkpoints.
  Iteration 20's halfway rule correctly failed on its own runs (GIVE seed 42/43 gaps -0.003 and +0.016 at checkpoint
  14). The Iteration 22 trajectories show that such a rule could miss gaps that appear later. They don't reveal what
  the stopped Iteration 20 runs would have reached.
- Females go the other way. GIVE female gaps are negative (-0.11 to -0.36), INERT female gaps are about zero: females
  learn to request the gift bit *less* when a gift could reach their mate. Why is open. Female eligible states are rare
  in GIVE (as low as 4% of female steps in seed 44), so the female gap rests on few states.
- GIVE males are eligible more often (p_eligible 0.55-0.66 vs 0.45-0.51 for INERT). That may mean they stay closer to
  the mate, or that mates need energy more often. The gap itself is a within-arm contrast, so it is not just this share.
- Mean predator counts are higher in GIVE (24.4-26.9 vs 19.1-22.3).

**Caveats.**

- Data-informed follow-up: the male focus and this comparison were chosen after Iteration 21 showed the w = 0 male
  gap, and this line had been closed under Iteration 20's hard stop. The user explicitly reopened it for this one
  control round. An independent replication with new seeds would make it a confirmatory result.
- The arms were trained in different batches (GIVE 2026-11-16 afternoon, INERT that night), with the same code version
  (a1854b09) and seeds. Iteration 21 showed about 0.05 run-to-run variance on this gap at equal seeds. The smallest
  male difference here (+0.044, seed 43) is within that range; the other two are well outside it.
- H2 does not show that the selectivity causes the birth gain. Untargeted transfers alone raise births (Iteration 19
  step-0 screen), and GIVE transfers energy in every state where a gift is valid, selective or not. The birth difference
  is consistent with provisioning paying, but this design cannot separate the targeted part from the transfers as such.
- Eligibility bundles mate within 3 cells, mate's energy below the reproduction threshold, and both able to afford
  the gift. The gap does not say which part the requests track, and each part may correlate with other policy inputs. Range alone may carry much of it.
  Eligibility is read before movement, the transfer after it.
- Three seeds.

**Status.** The mate-provisioning line has its first positive result on its registered test: male gift requests
associated with gift eligibility, with no added mate-regard reward (the predators' rewards are those of Iteration 20), against a matched
non-transferring control. What drives the learning has not been tested. The open next steps,
none of them pre-registered, are: a replication with new seeds, trained GIVE and INERT in one batch; a decomposition
of the gap by range versus mate energy; and the female sign reversal.

## Iteration 23 — Same-batch replication with new seeds (pre-registered 2026-11-17, before any training)

**Why.** Iteration 22's positive result (H1 + H2) was a data-informed follow-up, with the two arms trained in different
batches. This iteration replicates it confirmatorily: new seeds, both arms trained in one batch, tests fixed before
any of these runs exist.

**Design.** Six runs `REP23_{GIVE,INERT}_K05_MB1024_SEED{45,46,47}` (`compound_gift=on|inert`), both trained by this
module's tune script. Configuration as in Iterations 20-22: mate channel on, gift amount 2.0, no mechanical gift,
PROP k = 0.5, minibatch 1024, 30 epochs, 300 iterations. The GIVE arm is identical to Iteration 22's GIVE runs
(Iteration 21's w = 0 env, which is reward-identical by test). Code version: 76295b4e (env and analysis unchanged since
a1854b09). Waves of three, interleaved so each wave has both arms: wave 1 GIVE 45, INERT 45, GIVE 46; wave 2 INERT 46,
GIVE 47, INERT 47. Final analysis: checkpoint 29, 20 episodes, seeds 7000-7019, `analyze_compound_gift.py`. Launcher:
`run_iteration23.sh`, output `runs/analysis23/`.

**Pre-registered tests** (as in Iteration 22):

- **H1 (males).** In all three seeds, the GIVE male gift-bit gap is >= 0.02 with its 95% interval excluding zero, AND
  exceeds the matched-seed INERT male gap by >= 0.02.
- **H2.** GIVE births per 1000 predator-steps exceed INERT's in all three matched pairs.
- **Verdict.** H1 + H2: Iteration 22 replicated. H1 alone: the male eligibility association replicated, the birth
  difference did not. H2 alone, or neither: not replicated. Female gaps (Iteration 22: opposite sign), gift and request
  rates, eligibility shares and predator counts are descriptive. The interpretation limits of Iteration 22 still apply
  (observational state contrast, ecology differs between arms, birth gain not attributable to the selectivity).

**Halfway kill rule** (lenient on purpose: Iteration 22 showed the gap grows between checkpoints 14 and 29). At
checkpoint 14 of wave 1, analyse GIVE 45, GIVE 46 and INERT 45 (10 episodes, seed 7000). Continue unless neither GIVE
run's male gap exceeds INERT 45's (difference > 0).

**Hard stop.** One round, no changes after seeing any result.

**Planned next, after this iteration (not part of its tests).** Split the male gap into its parts: mate within
range versus mate's energy need. CPU only, on existing checkpoints. Requested by the user on 2026-11-17.

**Amendment, 2026-11-18 02:05 (wave 2 running, no wave-2 checkpoint 14 yet).** Wave 1 passed its halfway rule at
00:18 (GIVE minus INERT 45 male gaps +0.034 and +0.019 at checkpoint 14). Two decisions added with the user, before any
wave-2 data exist:

- **Wave-2 futility check** (`futility_iteration23_wave2.sh`, runs beside the launcher). At checkpoint 14 of INERT 46,
  GIVE 47 and INERT 47, analyse the pairs 46 and 47 (10 episodes, seed 7000; GIVE 46 uses its existing checkpoint 14).
  If either pair's GIVE male gap is not above its INERT gap, stop training: H1 needs all three seeds, so the iteration
  is scored H1 not met. This can only end the run early, never make a pass easier.
- **Hard stop for the mate-provisioning line.** If Iteration 23 does not replicate (verdict other than H1 + H2), run
  the planned range-versus-energy decomposition on existing checkpoints and then close the line: no Iteration 24.

**Results (2026-11-18): H1 and H2 met; by the registered verdict, Iteration 22 replicated.** With new seeds and both
arms in one interleaved two-wave launch, trained GIVE males' gift requests are again associated with pre-movement gift eligibility
far more strongly than INERT males' requests, and GIVE runs again have more births. The interpretation limits of
Iteration 22 still apply (below).

All six runs completed 300 iterations (exit 0). Wave 1's halfway check passed at 00:18 (GIVE minus INERT 45 male gaps
+0.034, +0.019); the wave-2 futility check passed at 03:37 (pairs 46, 47: +0.015, +0.048); the final analysis
finished at 05:19. Sources: `runs/analysis23/final.json`, `final.log`, `halfway.json`, `futility_wave2.json`,
`run_all.log`.

Checkpoint 29, 20 episodes, 95% episode-cluster bootstrap intervals:

| seed | GIVE male gap | INERT male gap | male diff | GIVE female gap | INERT female gap | births GIVE | births INERT |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 45 | +0.061 [0.054, 0.069] | -0.013 [-0.019, -0.007] | +0.074 | -0.062 | +0.013 | 3.13 [2.89, 3.34] | 2.92 [2.77, 3.06] |
| 46 | +0.057 [0.050, 0.062] | -0.012 [-0.018, -0.005] | +0.068 | -0.087 | -0.002 | 3.26 [3.06, 3.44] | 2.88 [2.69, 3.06] |
| 47 | +0.112 [0.102, 0.121] | -0.009 [-0.015, -0.003] | +0.121 | -0.122 | -0.048 | 3.49 [3.26, 3.74] | 2.82 [2.61, 3.01] |

- **H1 met.** Each GIVE male gap is >= 0.02 with its interval excluding zero, and exceeds the matched INERT gap by
  >= 0.02 (+0.068 to +0.121). INERT male gaps are slightly negative (-0.013 to -0.009). As in Iteration 22, this is an
  observational state contrast between arms whose ecology differs, not a manipulation of eligibility.
- **H2 met.** GIVE births exceed INERT's in all three pairs (one-sided exact sign test p = 0.125; a directional
  criterion, not conventional significance). In seed 45 the two intervals overlap.

**Descriptive.**

- Across Iterations 22 and 23, the male difference is positive in 6 of 6 seed pairs (+0.044 to +0.149), and the
  smallest difference here (+0.068) is above the ~0.05 run-to-run variance seen at equal seeds in Iteration 21.
- The GIVE male gap again grew from checkpoint 14 to 29 in every run (checkpoint 14: +0.030, +0.015, +0.045 for seeds
  45, 46, 47); checkpoint 14 used 10 episodes, with no uncertainty
  analysis of the change.
- Females again go the other way: GIVE female gaps -0.06 to -0.12, INERT -0.05 to +0.01. Smaller than Iteration 22's
  (-0.11 to -0.36). Why is open.
- GIVE males are eligible more often (p_eligible 0.59-0.68 vs 0.47-0.51) and request the gift bit somewhat more
  overall (527-568 vs 476-518 per 1000 male steps). Completed gifts: 19.2-19.6 per 1000 predator-steps.
- Mean predator counts are *not* higher in GIVE this time (21.6, 25.5, 21.6 vs INERT 25.2, 22.2, 24.3), unlike
  Iteration 22 (24.4-26.9 vs 19.1-22.3). Births are per 1000 predator-steps, so the contrast is not merely GIVE accumulating more
  predator-steps. That doesn't rule out population composition, density or other ecological differences as
  explanations for the rate difference; why population size differs between iterations is open.

**Caveats.**

- The H2 caveat of Iteration 22 is unchanged: GIVE transfers energy in every state where a gift is valid, and
  untargeted transfers alone raise births (Iteration 19 step-0 screen), so the birth gain can't be attributed to the
  selectivity.
- Eligibility still bundles mate within 3 cells, mate's energy below the reproduction threshold, and both able to
  afford the gift. Which part the requests track is untested; range alone may carry much of it. This is the planned
  decomposition below.
- Three seeds per iteration; six pairs in total, from two batches.

**Status.** Iteration 22 met the pre-registered replication verdict, with new seeds and GIVE and INERT in one
interleaved two-wave launch. The H1/H2 criteria were fixed before training. After wave 1 passed, a wave-2 futility
stop was added before any wave-2 checkpoint-14 data existed; it could only stop early, not make a pass easier.
The hard stop agreed on 2026-11-18 (close the line if this failed) did not trigger. Next is the
planned range-versus-energy decomposition (CPU only, existing checkpoints); any further training needs a new,
separately agreed plan with its own budget.

## Decomposition: range versus mate need (pre-registered 2026-11-18, before running)

**Why.** The eligibility gap bundles three conditions: mate within 3 cells, mate's energy below the reproduction
threshold, and both able to afford the gift. A male could produce the gap by tracking range alone, which needs no
regard for the mate's state. This analysis asks which part the requests track. Requested by the user on 2026-11-17.
It uses existing checkpoints only, no training.

**Design.** `analyze_gift_decomposition.py`, checkpoint 29, 20 episodes, seeds 7000-7019, on the six GIVE/INERT
pairs: Iteration 22 (GIVE = `predator_mate_regard/runs/MATEREGARD_W0_*_SEED42-44`, INERT =
`COMPOUND_INERT300_*_SEED42-44`) and Iteration 23 (`REP23_*_SEED45-47`). Each predator decision state (read before
the step) goes into one bucket: no mate; mate dying (energy minus homeostatic cost <= 0); giver can't afford; or a
2 x 2 cell of range (within 3 cells) x need (mate energy below the threshold). `in_need` is exactly the eligible
state (tested). The mate's energy is observable: the predator layer holds energy values and the mate channel marks
the mate's cell, and a mate within 3 cells is always inside the 7 x 7 window. Contrasts, per episode and sex:

- `need_in` = P(gift bit | in range, mate needs) - P(gift bit | in range, mate full)
- `range_need` = P(gift bit | in range, mate needs) - P(gift bit | out of range, mate needs)

**Pre-registered tests (males).**

- **D1, need tracking.** In at least 5 of 6 pairs: the GIVE `need_in` is >= 0.02 with its 95% episode-bootstrap
  interval excluding zero; GIVE minus matched INERT is >= 0.02 with the 95% interval of that difference (independent
  episode bootstraps) excluding zero; and both runs have the contrast defined (both cells present) in >= 10 of the
  20 episodes, else the pair fails. Scored by `score()` in the script.
- **D2, range tracking.** The same rule for `range_need`.
- **Verdict.** D1 + D2: requests track both. D2 only: the gap is explained by the mate being in range, with no
  evidence of regard for the mate's need. The gift range equals the observation radius (3), so an in-range mate is
  always tagged and visible and an out-of-range mate never is: D2 can't separate range from seeing the mate. D1 only: need, not range. Neither: the gap comes from the combination or from states these cells
  don't separate. Females, bucket shares and the other cells are descriptive.
- **My prediction:** D2 met, D1 not met (range carries the gap).

**Limits.** Same observational caveats as Iterations 22-23: the arms differ in ecology, and "mate full" states
within range are a selected set (a mate above the threshold may be about to reproduce, or may just have received
a gift). A pass on D1 shows the requests are associated with the mate's energy, not that males perceive or value it.

**Amendment before running (Codex review, 2026-11-18).** The minimum-episode rule, the interval on the GIVE-minus-INERT
difference and the range/visibility qualification above were added after a Codex review of the script and this
pre-registration, before any decomposition output existed. Boundary tests for the buckets were added at the same time.

**Hard stop.** One analysis run, no changes after seeing results. If neither D1 nor D2 passes, no further
decomposition. Any further training needs a separately agreed plan.

**Results.** (not yet run)
