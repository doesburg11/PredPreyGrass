# predator_compound_provisioning

> ## STATUS: CLOSED 2026-11-16, NOT MET (Iteration 20 stopped by its halfway kill rule)
>
> At iteration 150 the gift bit was still about a coin flip (47-53% of steps) in both arms. GIVE's selectivity gap was
> never more than 0.001 above INERT's (the rule needed 0.02), so wave 2 was never started. GIVE had more births, but that
> comes from random transfers, as in Iteration 19. The mate-provisioning line is closed. Details are in
> [`RESULTS.md`](RESULTS.md).
>
> **Original question:**
>
> Do sexed predators learn to feed their mate selectively when the gift can help her or him reach the
> reproduction threshold? This follows Iteration 19 of
> [`predator_sexual_reproduction`](../predator_sexual_reproduction/RESULTS.md), where a learned gift action produced more
> births but the selectivity test failed in one of three seeds. There, the gift was also a second "stand still" action.
> Here every move comes in two versions, with and without a gift request (18 predator actions), and a matched INERT
> arm has the same actions but never transfers. Design, tests, halfway kill rule and hard stop are in
> [`RESULTS.md`](RESULTS.md).

## Files

- `compound_gift_env.py`: the 18-action environment (subclass of the `predator_sexual_reproduction` env) and its RLlib env creator.
- `config_env.py`: the module config (compound gift on, mate channel on, mechanical gift off).
- `tune_ppo_compound_provisioning.py`: training entry point; reuses `predator_sexual_reproduction`'s PPO script
  (`--compound-gift on|inert` plus that script's options). Runs land in `runs/` (gitignored).
- `analyze_compound_gift.py`: gift-bit selectivity, gift rates and pair outcomes from a checkpoint.
- `run_iteration20.sh`: staged launcher for the pre-registered six runs, with the halfway kill check.
- `tests/`: environment and analysis tests. Run with
  `pytest predpreygrass/non_evolutionary/discrete_space/predator_compound_provisioning/tests`.
