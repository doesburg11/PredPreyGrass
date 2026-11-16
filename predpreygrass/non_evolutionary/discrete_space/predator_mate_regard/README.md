# predator_mate_regard

> ## STATUS: CLOSED 2026-11-16, NOT MET (Iteration 21)
>
> With w = 0.5, males became more selective than matched w = 0 runs in only 1 of 3 seeds, and births were higher in only
> 1 of 3 pairs. Fresh w = 0 runs develop a similar male gap (+0.06 to +0.14) without any motivation. The halfway pass
> came from an unusually low reused reference run. Steps 2-3 not built. Details are in [`RESULTS.md`](RESULTS.md).
>
> **Original question:**
>
> Can a motivation to value the mate's energy intake make PPO predators learn to feed their mate selectively?
> Iteration 20 ([`predator_compound_provisioning`](../predator_compound_provisioning/RESULTS.md)) showed that gifts raise
> births but are never learned: the payoff arrives too late, through the mate's reproduction. Here a predator's reward
> includes w times its mate's energy reward (w = 0.5 vs w = 0). This is step 1 of 3. Step 2 is a payoff check. Step 3,
> only if both pass, makes w heritable. Design, tests, halfway kill rule and hard stop are in [`RESULTS.md`](RESULTS.md).

## Files

- `mate_regard_env.py`: mate-regard reward on the 18-action compound-gift env, and its RLlib env creator.
- `config_env.py`: module config (compound gift on, mate channel on, `mate_regard_weight`).
- `tune_ppo_mate_regard.py`: training entry point (`--mate-regard-weight W` plus the core PPO script's options). Runs
  land in `runs/` (gitignored).
- `run_iteration21.sh`: staged launcher for the six pre-registered runs, with the halfway kill check.
- Analysis: `../predator_compound_provisioning/analyze_compound_gift.py`.
- `tests/`: run with `pytest predpreygrass/non_evolutionary/discrete_space/predator_mate_regard/tests`.
