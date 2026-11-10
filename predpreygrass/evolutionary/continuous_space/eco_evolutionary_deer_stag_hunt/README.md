# eco_evolutionary_deer_stag_hunt

**Status (2026-11-10): section 2 pre-registered and built; its pilot is next.**
Section 1 (2026-11-08, deer frozen at deer_stag_hunt's final policy) failed its pilot:
hunters died out in 50 of 50 episodes in every arm, likely because naive hunters met
trained deer ([RESULTS.md](RESULTS.md) §1.5). Section 2 ([DESIGN_2.md](DESIGN_2.md),
[RESULTS.md](RESULTS.md) §2) replays deer_stag_hunt's touch-rule deer in training
order, so hunters and deer start naive together, in the touch world.

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

- **Deer, section 2:** replayed from deer_stag_hunt's touch_((k − 1) mod 3) + 1 in
  training order, never trained here: iteration t uses that run's deer of iteration
  10·floor(t/10) (iteration 10 before 20), copied into each run with a hashed
  manifest (`runs/<run>/deer_schedule/`, [deer_schedule.py](deer_schedule.py)).
  Section 1 used one frozen final deer policy (`runs/<run>/deer_policy/`).
- **Same world:** without the gene inputs, an episode is deer_stag_hunt's for the same
  actions ([tests/test_equivalence.py](tests/test_equivalence.py)).

## Running

From the repository root, long jobs under `systemd-run --user`:

```bash
# Section 2 pilot: L on all three deer schedules, B on touch_1 (500 iterations), E matched, evaluations, check
systemd-run --user --unit eco-deer-pilot2 bash $PWD/predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/run_pilot_2.sh

# Section 2 main runs (only after the pilot passed): L_k, B_k (k = 1..6), E_k matched, evaluations, verdict
systemd-run --user --unit eco-deer-main2 bash $PWD/predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/launchers/run_main_2.sh

# One run of arm L or B, run k (seed k, deer replayed from deer_stag_hunt touch_((k-1) mod 3) + 1)
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
  - `run_pilot_2.sh`, `run_main_2.sh`, `pilot_check_2.py`, `verdict_2.py`: section 2's
    scripted pilot, main runs, pilot bar and verdict (section 1's scripts were removed
    after its stop; they are in git history, and `verdict.py` remains as the source of
    the shared run values).
- `tests/`: the genes and bias, evolution, the RL module, deer_stag_hunt's rules, and
  equivalence with deer_stag_hunt.
- `deer_schedule.py`: the replayed deer of section 2.
- `DESIGN.md`, `DESIGN_2.md`, `RESULTS.md`: the designs and the experiments.
