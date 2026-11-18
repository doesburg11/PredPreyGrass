# predator_compound_provisioning

> ## STATUS: Iteration 23 (2026-11-18): REPLICATED (H1 + H2 met, confirmatory)
>
> New seeds 45-47, GIVE and INERT in one interleaved two-wave launch. H1/H2 were fixed before training; a wave-2
> futility stop added after wave 1 passed (before any wave-2 halfway data) could only stop early. GIVE males' gift requests are again
> associated with gift eligibility far more strongly than INERT's (+0.068 to +0.121 above INERT, 3/3 seeds), and
> GIVE runs again have more births (3/3 pairs). Across Iterations 22-23 the male difference is positive in 6/6 seed
> pairs. Females again show the opposite sign. Same interpretation limits as Iteration 22: observational state
> contrast, ecology differs between arms, birth gain not attributable to the selectivity. Next: range-versus-mate-energy
> decomposition on existing checkpoints. Details in [`RESULTS.md`](RESULTS.md).
>
> ## Earlier status: Iteration 22 (2026-11-17): H1 + H2 MET, data-informed follow-up
>
> With the INERT control trained to 300 iterations, GIVE males' gift requests are associated with gift eligibility far
> more strongly than INERT's (gap +0.044 to +0.149 above INERT, 3/3 seeds), and GIVE runs have more births (3/3 pairs).
> The gap was larger at iteration 300 than at 150 in all three runs, so Iteration 20's halfway rule could miss
> late-appearing gaps. Females show the opposite sign. Caveats are in [`RESULTS.md`](RESULTS.md): data-informed, two
> training batches, an observational state contrast with ecology differing between arms, and a birth gain that can't
> be attributed to the selectivity. Open next steps are there too.
>
> ## Earlier status: CLOSED 2026-11-16, NOT MET (Iteration 20 stopped by its halfway kill rule)
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
- `run_iteration20.sh`, `run_iteration22.sh`, `run_iteration23.sh`: staged launchers with halfway kill checks;
  `futility_iteration23_wave2.sh`: Iteration 23's added wave-2 futility check.
- `tests/`: environment and analysis tests. Run with
  `pytest predpreygrass/non_evolutionary/discrete_space/predator_compound_provisioning/tests`.
