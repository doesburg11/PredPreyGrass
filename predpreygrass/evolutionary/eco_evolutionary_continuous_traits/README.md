# eco_evolutionary_continuous_traits

**Status (2026-10-17): pilot and v2 round done** ([RESULTS.md](RESULTS.md)).
With expensive capacity (pilot) the traits evolved by selection into cheap
bodies and the ecosystem and the learning arms race collapsed. The two
cheaper-capacity runs (v2: 2× and 4× cheaper, also with fixed observation
scales) were stable (coexistence 50 of 50); both species traded view
distance for a wide field of view, speed rose and fell again, and newer
prey checkpoints did better while newer predator checkpoints mostly did
worse. One run per condition.

Heritable body traits in the continuous Predator-Prey-Grass world: does an
eco-evolutionary arms race appear when animals' bodies evolve while their
behaviour is learned?

[`base_environment_continuous`](../../non_evolutionary/continuous_space/base_environment_continuous/)
showed a learning arms race in that world: one shared PPO policy per
species, newer predators catching fixed prey better and newer prey evading
fixed predators better, in five independent runs, with and without
Aquarium's quirks (its RESULTS.md, sections 2 and 3). That is co-adaptation
by learning only: no genomes, no heritable variation, no selection. This
module, a copy of that one, adds evolution: every animal carries a genome
of body traits, and its offspring inherit it with mutation.

The project's goal is a sustainable, significant Darwin/Baldwin loop in
predator-prey coevolution. The closest earlier result, ERL coevolution
(Trial 14,
[`eco_evolutionary_erl_coevolution`](../eco_evolutionary_erl_coevolution/)),
evolved per-agent networks with a weak lifetime learner on a grid and
produced no arms race. Here the learner is the PPO policy that does produce
one, and evolution acts on bodies.

## Design

Full rules in [SPEC.md](SPEC.md), section 11.

- **Genes:** top speed, view distance and field of view, as log-ratios to
  the species' base values (×0.5 to ×2).
- **Cost:** a larger capacity raises the resting metabolic cost (speed
  1.0, vision 0.5, field of view 0.25 per unit of relative change); moving
  still costs per unit of speed. The coefficients are starting guesses.
- **Inheritance:** asexual, with mutation; each episode's founders are drawn
  from the survivors of the previous one, so the gene pool evolves across
  PPO episodes. Each simulation worker evolves its own population.
- **Learning:** one shared PPO policy per species, which sees each animal's
  own genes (3 extra inputs) and can learn to use the body it has.
- **World:** every quirk of SPEC.md section 9 fixed (the world of BE-04
  and BE-05).
- **Neutral control:** `trait_expression = False` carries the genes the same
  way without effect, measuring drift.

What would count as a result, in order:

1. **Selection, not drift:** within every run, selection differentials
   (covariance of each gene with relative offspring count, logged per
   episode) consistently away from zero in expressed runs and around zero
   in neutral runs, with trait trajectories that follow them. The neutral
   control alone is not enough: expressed traits change population sizes,
   so the two conditions drift differently.
2. **Coevolution:** a predator trait change followed by a matching prey
   change (or the reverse), e.g. a speed or vision race, in addition to the
   learning arms race.
3. **Baldwin interaction:** learning changing the course of evolution, e.g.
   traits evolving differently with a frozen policy than with a learning
   one.

**Units of replication.** Each simulation worker evolves its own gene pool,
but all workers train one shared policy per species, so a policy change
learned on one island changes fitness on every other: workers are not
independent replicates. Independent training runs are.

**Known limitation:** each episode restarts from 6 predators and 8 prey
drawn from the survivors of the last one, a tight bottleneck between
episodes that strengthens drift.

## Files

- [`environment.py`](environment.py): the world with traits.
- [`callbacks.py`](callbacks.py): logs each species' mean and spread of
  every gene, cost multiplier, numbers and births at the end of every
  episode (TensorBoard tags `env_runners/traits/...`).
- [`train.py`](train.py): PPO training; prints the mean genes per species
  every iteration and saves every worker's gene pool with each checkpoint.
- [`checkpoints.py`](checkpoints.py): loads a checkpoint's policies and gene
  pool.
- [`analysis/pilot.py`](analysis/pilot.py): the pilot's figure and
  results JSON (CT-01 against CT-02).
- [`analysis/v2_costs.py`](analysis/v2_costs.py): the v2 round's figure and
  results JSON (CT-03 to CT-05), with bootstrap intervals for the
  tournaments.
- [`analysis/final_check.py`](analysis/final_check.py),
  [`tournament.py`](tournament.py): evaluation with each checkpoint's
  evolved gene pool.
- [`config/`](config/): settings (traits and quirk fixes in
  `config_env.py`).
- [`tests/`](tests/): traits (`test_traits.py`, including bit-for-bit
  equivalence with `base_environment_continuous` when traits are off),
  quirk fixes and the environment.

Unchanged copies from `base_environment_continuous`: `geometry.py`,
`grass.py`, `render.py`, `eval.py`, `assets/`.

## Usage

From the repository root:

```bash
python predpreygrass/evolutionary/eco_evolutionary_continuous_traits/train.py
python -m pytest predpreygrass/evolutionary/eco_evolutionary_continuous_traits/tests
```

Runs are saved in the module's `runs/` folder (gitignored).
