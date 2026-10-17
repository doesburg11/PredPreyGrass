# Predator-Prey-Grass drive-conditioned environment

An **observation-representation experiment** derived from
[`base_environment_discrete`](../base_environment_discrete/README.md): does giving
PPO explicit summaries of hunger, reproductive readiness, and local resources
help learning, without changing the ecological rules or reproduction-only reward?

**Status:** the full feature set is implemented and one matched baseline-versus-drive
comparison has completed. It reports a **+6.0% episode-weighted return across
training**, mainly from predator reproduction. This is a single training seed per
arm, not an established advantage across seeds. The energy-only comparison and
stronger tactical-feature experiments remain untested.

## What changes

The baseline supplies four local spatial channels: borders, predator energy,
prey energy, and grass energy. **An agent's own energy is already represented at
its position in the relevant energy channel.** Drive conditioning makes selected
information explicit and normalized; it does not give agents their first access
to their own energy.

The full feature set adds the following scalars, clipped to `[0, 1]`:

| Feature | Species | Calculation |
| --- | --- | --- |
| `hunger_pressure` | Both | `1 - own_energy / hunger_safe_energy` |
| `reproductive_readiness` | Both | `own_energy / reproduction_threshold` |
| `prey_opportunity` | Predator | Sum of local prey-energy channel / configured normalizer |
| `predator_danger_pressure` | Prey | Sum of local predator-energy channel / configured normalizer |
| `grass_opportunity` | Prey | Sum of local grass-energy channel / configured normalizer |

The density-based features summarize **energy in the observation window**, not
literal counts of nearby animals or grass patches. A danger feature is therefore
a local exposure proxy, not a calculated probability of being caught.

Each scalar is broadcast as a constant channel across the local window. Predators
receive **7 channels** (4 spatial + 3 derived); prey receive **8** (4 + 4).
The `energy_only` option gives each species only hunger and reproductive readiness,
for 6 channels each. Feature lists and normalizers are defined in
[`config_env.py`](config_env.py); the formulas are implemented in
[`predpreygrass_rllib_env.py`](predpreygrass_rllib_env.py).

Rewards remain **+10 per birth**, with other reward terms zero in the experiment.
Actions and ecological rules are unchanged. The same CNN design is used, but its
input channel count—and consequently the first convolution's parameter count—
changes. This comparison is not parameter-count matched.

No feature supplies a direction, target, or scripted response. The policy still
learns how to act. Names such as “hunger” describe environment-computed inputs;
they do not demonstrate an internal motivational state learned by the agent.

## Why this experiment is useful

The energy features provide a convenient representation relative to explicit
thresholds; the spatial summaries provide aggregate context that a network could
otherwise derive from the grid. Either could reduce the learning needed to use
those inputs, but that mechanism has not been isolated experimentally.

This differs from [`project_reward_shaping`](../project_reward_shaping/README.md):
that investigation changes the reward, while this one changes observations.
Leaving reward unchanged preserves the reproduction objective. It does not
guarantee better optimization or transfer to new environments.

## Completed comparison: one training seed per arm

Both arms used seed 42 and 1,000 PPO iterations with matching ecological settings
and PPO hyperparameters. The observation channels differed as described above.

- Baseline: `PPO_BASE_ENVIRONMENT_SEED42_2026-09-05_18-55-45`.
- Full drives: `PPO_DRIVE_CONDITIONED_FULL_SEED42_2026-09-06_07-26-38`.

The archived comparison is identified as `base_v_drive_2026-09-06` in local run
storage. Baseline outputs are stored with the base module; full-drive outputs
are stored with this module. Both runs completed all training iterations.

The recorded analysis pools returns by completed-episode count, rather than
averaging iteration means equally: iterations can finish different numbers of
episodes, including none.

| Across-training measure | Baseline | Full drives |
| --- | --- | --- |
| Completed episodes | 1,055 | 1,057 |
| Episode-weighted return | 6,905.1 | 7,322.6 |
| Relative return difference | — | +6.0% |

With `return = 10 × (predator_births + prey_births)`, this corresponds to about
6% more combined births per completed episode **averaged across training**.
It is not a standalone estimate of the final checkpoints' performance, a 6%
wall-clock speedup, or a 6% increase in surviving population.

**Shape of the effect over training** (episode-pooled return, gap vs. baseline):

| Iterations | Gap |
| --- | --- |
| 1–25 (both still learning to survive) | +76.0% |
| 26–50 (both just reached full-length episodes) | +28.9% |
| 51–100 | +9.3% |
| 101–200 | +4.4% |
| 201–300 | +6.0% |
| 301–400 | +6.5% |
| 401–500 | +6.6% |
| 501–600 | +3.1% |
| 601–700 | +2.8% |
| 701–800 | +5.3% |
| 801–900 | +5.3% |
| 901–1000 | +7.3% |

The observed advantage is largest early in training and remains positive in
later windows. Those windows belong to the same two runs; they are not independent
replications. The pattern is consistent with improved learning, but does not
identify its cause.

### Which species benefited?

The recorded analysis attributes **70.7% of the full-run return gap to predator
births**, and approximately all of the gap in iterations 501–1,000. Prey births
were only about 1–2% higher. Late-training end populations were similar between
arms, around 17–18 predators and 23–24 prey. Both runs generally sustained
coexistence, with reported extinction metrics near zero.

A prey-heavy population snapshot around iteration 332 did not persist in the
completed comparison. It should not be presented as a stable population effect.

### Feature calibration

Recorded channel diagnostics showed non-degenerate distributions rather than
constant saturation: mean hunger pressure 0.15, reproductive readiness 0.48,
prey opportunity 0.50, predator danger 0.55, and grass opportunity 0.39. This
supports their numerical usefulness in this run. It does not establish optimal
normalizers or generalization across ecological settings.

## What remains unresolved

1. **Replication:** at least several additional independent training seeds per
   arm are needed to assess whether the sign and size of the return difference
   persist. A common seed label does not make these different architectures
   follow identical random trajectories.
2. **Feature attribution:** the implemented `energy_only` arm has no completed
   comparison data. It can test whether the energy summaries account for most
   of the observed benefit or whether local-context summaries add value. A
   single additional run would remain exploratory rather than conclusive.
3. **Final-policy evaluation:** compare final checkpoints across multiple
   episode seeds, reporting births by species, extinction frequency, and
   population outcomes separately from across-training returns.
4. **Mechanism and controls:** the current comparison cannot separate input
   normalization, ecological summaries, extra parameters, and training
   variability. It does not establish reduced value-function variance or
   better transfer to different energy thresholds or resource settings.

Stronger features such as an escape direction or best hunting target have not
been tested here. They would supply tactical advice and should be treated as a
separate experimental extension.

## Running the experiment

From the repository root, with the project environment installed:

```bash
# Full feature set (the completed comparison arm).
python -m predpreygrass.non_evolutionary.discrete_space.drive_conditioned_environment.tune_ppo_drive_conditioned_environment --seed 42 --drive-set full --max-iters 1000

# Energy-only ablation (implemented; no completed comparison yet).
python -m predpreygrass.non_evolutionary.discrete_space.drive_conditioned_environment.tune_ppo_drive_conditioned_environment --seed 42 --drive-set energy_only --max-iters 1000
```

These commands start new training runs; they do not reproduce an analysis by
themselves. New results go under this module's `runs/` directory. For baseline
replications, use the [base trainer](../base_environment_discrete/tune_ppo_base_environment.py)
with matching settings and seeds. For interactive playback, set the checkpoint
path in the [evaluator](evaluate_ppo_from_checkpoint_debug.py) and keep the
observation feature configuration consistent with the trained checkpoint.
