# predator_mate_regard — results

Follow-up to [`predator_compound_provisioning`](../predator_compound_provisioning/RESULTS.md) (Iteration 20) and
[`predator_sexual_reproduction`](../predator_sexual_reproduction/RESULTS.md) (Iterations 0-19). Numbering continues.

## Iteration 21 — Mate-regard reward, step 1 of 3 (pre-registered 2026-11-16, before any training)

**Why.** Iterations 17-20 closed the mate-provisioning line. Transfers to the mate raised births whenever they happened,
and the Iteration 19 step-0 screen showed that giving pays the giver. Yet PPO never learned to target gifts (Iteration 20:
the gift bit stayed at about 50% in all states). The payoff exists but comes many steps later, through the mate's
reproduction. The working hypothesis is a credit-assignment barrier, not a missing payoff. Humans may bridge such
delays partly through evolved motivations to value others' welfare. This line tests that idea in three steps, each
with its own stop rule:

1. **(this iteration)** Can a fixed motivation of this kind make PPO learn selective giving at all?
2. Payoff check: do predators with the motivation out-reproduce those without, in a shared world?
3. Only if both pass: make the weight heritable and test whether it rises from near zero.

**Mechanism.** Each predator's reward gains `w * reward_predator_per_energy * intake(mate)`: w times the energy reward
its recorded mate earns this step, where intake is food eaten plus gifts received. With w = 0.5 a predator values its
mate's energy intake at half its own. The mate is read from the bond at the start of the step, the bond a gift in that
step goes to. A predator that dies in a step still gets the bonus for its mate's intake in that step (its terminal
transition), and the truncation call at max_steps carries no bonus. Dynamics, actions and observations are Iteration 20's 18-action compound gift (`compound_gift="on"`).
Only rewards change. Note: with w > 0, a valid gift yields the giver an immediate bonus of at least w x 0.5 x 2 = 0.5
at w = 0.5 (more if the mate also eats in that step), so this step tests a motivation that pays within the step by construction. A pass shows that the motivation
can be learned, not that cooperation pays. Step 2 tests the payoff.

**Code.**

- `predator_sexual_reproduction/predpreygrass_rllib_env.py`: bookkeeping only. `predator_energy_intake` records each
  predator's energy intake per step (fruit, prey, mechanical and learned gifts received). It never changes dynamics or
  rewards.
- `mate_regard_env.py`: `MateRegardPredPreyGrass` (subclass of the compound-gift env) and the remote-runner env
  creator. Episode metrics `ecology/mate_regard_weight` and `ecology/mate_regard_reward_total`.
- `tune_ppo_mate_regard.py`: `--mate-regard-weight W` plus the core PPO script's options.
- `run_iteration21.sh`: staged launcher. It includes the launcher fixes from the Iteration 20 Codex review: every PID
  is waited for, any failure stops all training, the final-analysis status is checked, and checkpoint readiness
  waits for `rllib_checkpoint.json` plus 2 minutes.
- Pre-launch Codex review (2026-11-16) found two launch-blocking bugs, both fixed before launch:
  - Stale intake was rewarded again on the all-truncated call at max_steps. Intake is now cleared at the top of
    `step`, and a regression test covers it.
  - The halfway kill rule rounded the difference before comparing it with 0.02. It now compares the raw value.
  Minor points also fixed: the stop path exits directly, letting systemd clean up the cgroup; the terminal-transition
  bonus is documented; the gift bonus is stated as a minimum.
- Analysis: `predator_compound_provisioning/analyze_compound_gift.py`, with two amendments from the Iteration 20 Codex
  review, made before any Iteration 21 result exists:
  - "Eligible" now charges the homeostatic cost: the mate's energy must stay positive after it, and the giver must keep
    positive energy after it plus the gift (stay-put variant).
  - Per-sex request rates are reported.
- Tests: `tests/` (7) plus the compound (21) and `predator_sexual_reproduction` (175) suites, 203 in all, passing. A
  test shows that w = 0 reproduces the compound env's rewards and terminations exactly over a random rollout.
- Smoke run: 4 iterations with 2 remote env runners, w = 0.5. Workers report `mate_regard_weight` 0.5 and a nonzero
  `mate_regard_reward_total` (24-40 per episode). The remote runners build the right env with the right weight.

**Design.** Six runs, `MATEREGARD_{W05,W0}_K05_MB1024_SEED{42,43,44}`, with w = 0.5 (W05) or w = 0 (W0). Everything
else is Iteration 20's GIVE configuration: compound gift on, mate channel on, gift amount 2.0, no mechanical gift,
PROP k = 0.5 (flat forage rewards 0, energy reward 0.5, combat-death penalty 0.2), minibatch 1024, 30 epochs,
300 iterations, three runs at a time (GPU fraction 0.33, 4 env runners x 2 CPUs). Wave 1 is W05 42/43/44.
Wave 2 (W0 42/43/44, trained fresh) starts only after the halfway check passes and wave 1 ends. Final analysis:
checkpoint 29 (iteration 300), 20 episodes, evaluation seeds 7000-7019.

**Pre-registered tests.** Thresholds are as in Iterations 19-20.

- **H1, learned selective giving.** For the same sex in all three seeds: the W05 gift-bit gap (eligible minus other
  states) is >= 0.02 with its 95% interval excluding zero, AND it exceeds the matched-seed W0 gap of that sex by >= 0.02.
- **H2, births.** Births per 1000 predator-steps are higher in W05 than W0 in all three matched pairs (complete paired
  separation; one-sided exact sign test p = 0.125. A stringent directional criterion, not conventional significance).
- **Verdict and next step.** Step 2 (payoff check) is built only if H1 is met. H1 + H2: the motivation is learnable and
  coincides with more births. H1 alone: learnable, no birth difference. H2 without H1: not attributable to selective
  giving. Not met (no H1): the line closes at step 1, because if a motivation that pays within the step can't produce
  selective giving, an evolved one has nothing to act on. Gift and request rates, which sex gives, proximity to the
  mate, population size and the mate-regard reward total (training metric `ecology/mate_regard_reward_total` in
  `result.json`; the evaluation analysis replays dynamics only) are descriptive only.

**Halfway kill rule** (efficiency rule, not a result test). When checkpoint 14 (about iteration 150) exists for
W05 42/43/44, run the analysis on those three and on Iteration 20's `COMPOUND_GIVE_K05_MB1024_SEED{42,43}`
checkpoint 14 as the w = 0 reference. That run has the same configuration with w = 0, and the w = 0 path is
reward-identical by test. Settings: 10 episodes, seed 7000, amended eligibility. Continue only if, for seed 42 or 43,
at least one sex has a W05 gap at least 0.02 above the matched W0 reference gap. Otherwise stop wave 1 and don't start
wave 2. The reference runs come from the earlier code version, so wave 2 retrains W0 fresh for the final test.

**Hard stop.** One round: these six runs and the analysis above. No change to the weight, rewards, actions, gift
amount, seeds, duration, thresholds or analysis after seeing any result, including the halfway read. If H1 is not met,
the mate-regard line closes.

**Caveats fixed in advance.** Three seeds give no power for small effects. W05 changes the reward scale (extra positive
reward when the mate eats), so W05 and W0 differ in more than the gift incentive. W05 may, for example, learn to stay
near the mate or hunt near it. That is descriptive and may itself raise the eligible share. The gap compares eligible
with all other states, so it can come mostly from range (mate within 3 cells), not from the mate's energy need.
Eligibility is read before movement, the transfer after it.

**Results.** (not yet run)
