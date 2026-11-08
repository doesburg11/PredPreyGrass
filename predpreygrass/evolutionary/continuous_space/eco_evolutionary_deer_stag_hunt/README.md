# eco_evolutionary_deer_stag_hunt

**Status (2026-11-08): pilot FAILED (hunters extinct in 50 of 50 episodes in every arm;
likely cause: naive hunters against frozen trained deer); stage stopped, see
[RESULTS.md](RESULTS.md) §1.5. Pre-registered in
[RESULTS.md](RESULTS.md), section 1.**

Nature, nurture and both in a stag hunt. This is
[deer_stag_hunt](../../../non_evolutionary/continuous_space/deer_stag_hunt/)'s world,
where hunters learned to approach adult deer with a partner near more often than
chance, with heritable innate attractions added to the hunters. The question: do the
genes plus learning do better than learning alone, or the genes alone? The design and
its reasons are in [DESIGN.md](DESIGN.md).

## The genes and the three arms

Every hunter carries four genes in [-1, 1]:

| Gene | Effect |
| --- | --- |
| `a_hunter` | attraction (+) or avoidance (-) toward hunters in view |
| `a_young` | the same toward young deer, fawns and middle, in view |
| `a_adult` | the same toward adult deer in view |
| `neutral` | none |

- **Inheritance:** a newborn inherits its parent's genes plus a small mutation.
- **Founders:** each episode's founders are the hunters alive at the end of that
  environment's previous episode, or its recent births after an extinction.
- **Gene inputs:** a hunter observes its own genes.
- **The bias:** the genes add `β · s_m · (u_m · d)` to each move's logit, where `d`
  is the gene-weighted mean direction to the hunters, young deer and adult deer in
  view ([genes.py](genes.py)).

| Arm | Moves from | Code |
| --- | --- | --- |
| E, evolution only | the bias alone (stops masked; straight ahead when nothing is in view) | [evolve.py](evolve.py) |
| L, learning only | PPO; genes inherited but not expressed | [train.py](train.py), β = 0 |
| B, both | PPO plus the bias, inside the module's forward passes | [train.py](train.py), [rl_module.py](rl_module.py) |

- **Frozen deer:** in all arms the deer use a deer_stag_hunt run's final deer policy,
  copied into each run (`runs/<run>/deer_policy/`).
- **Same world:** without the gene inputs, an episode is deer_stag_hunt's for the same
  actions ([tests/test_equivalence.py](tests/test_equivalence.py)).

## Running

From the repository root, long jobs under `systemd-run --user`:

```bash
# Pilot: L and B (500 iterations), E matched to them, evaluations
bash predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/run_pilot.sh

# Main runs: L_k, B_k (k = 1..5), E_k matched to them, evaluations
bash predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/run_main.sh

# One run of arm L or B, run k (seed k, deer of deer_stag_hunt run ((k-1) mod 3) + 1)
python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/replicate.py NAME B 1 [ITERATIONS]

# One E run matched to L and B runs
python -m predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.evolve NAME K --match RUN_DIR ...

# Evaluate a run at its marks; the tests and verdict
python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/evaluate_run.py NAME [MARK ...]
python predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/verdict.py
```

## Files

- `environment.py`: the world with genomes, inheritance, founder pools, selection
  differentials and lifetime counters.
- `genes.py`: the bias, the β₀ rule and E's move rule.
- `rl_module.py`: the predators' PPO module with the bias.
- `train.py`: arms L and B. It saves `gene_pools.json` and `budget.json` with every
  checkpoint.
- `evolve.py`: arm E.
- `checkpoints.py`: reading any arm's checkpoints.
- `evaluate.py`, `launchers/coop_fair.py`: evaluation.
- `launchers/`:
  - `replicate.py`, `evaluate_run.py`, `verdict.py`: runs, evaluation and the
    verdict;
  - `run_pilot.sh`, `run_main.sh`: the scripted pilot and main runs.
- `tests/`: the genes and bias, evolution, the RL module, deer_stag_hunt's rules, and
  equivalence with deer_stag_hunt.
- `DESIGN.md`, `RESULTS.md`: the design and the experiment.
