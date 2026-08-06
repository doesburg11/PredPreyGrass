# Predator-Prey-Grass base environment — seasonal variant

Same mechanics as [`base_environment`](../base_environment/), plus one addition: grass regrowth
rate cycles between an "abundant" and a "scarce" phase over the course of an episode, instead of
staying flat.

## Seasonal grass-regrowth cycle

`energy_gain_per_step_grass` is multiplied by `season_high_multiplier` for
`season_length_steps` steps, then by `season_low_multiplier` for the next
`season_length_steps` steps, repeating for the rest of the episode (a square wave, computed
purely from the env's own per-episode step counter — see `_current_season_multiplier()` in
`predpreygrass_rllib_env.py`). At each step: `phase = (current_step // season_length_steps) % 2` —
phase 0 uses the high multiplier, phase 1 uses the low one.

Three config keys in [`config_env.py`](./config_env.py) control it:

| Key | Default | Meaning |
|---|---|---|
| `season_length_steps` | 40 | how many env steps each phase lasts (a full cycle is 80 steps) |
| `season_high_multiplier` | 1.5 | multiplier during the "abundant" phase |
| `season_low_multiplier` | 0.5 | multiplier during the "scarce" phase |

With the defaults, grass regrows at 0.06/step for steps 0-39, 0.02/step for steps 40-79,
0.06/step again for 80-119, and so on.

**Equivalence to `base_environment`:** setting *both* `season_high_multiplier = 1.0` and
`season_low_multiplier = 1.0` makes this module behave exactly like `base_environment` — the
multiplier always evaluates to 1.0 regardless of `current_step`/`season_length_steps`, so
`energy_gain_per_step_grass * 1.0` is byte-for-byte the same flat rate as the original
(covered by `test_season_disabled_reproduces_flat_baseline` in
[`tests/test_seasonal_grass_regrowth.py`](./tests/test_seasonal_grass_regrowth.py)). Setting
only *one* of the two multipliers to 1.0 does **not** reproduce the original — you'd still get
an alternating cycle, just asymmetric (e.g. flat-then-scarce instead of constant-rate).

Everything else — energy costs, reproduction thresholds, rewards, population sizes — is
unchanged from `base_environment`.

## Season-multiplier parameter sweep

[`run_season_multiplier_sweep.sh`](./run_season_multiplier_sweep.sh) trains 6 regimes
sequentially (single GPU), 500 iterations each, sweeping `season_high_multiplier` /
`season_low_multiplier` from `1.0/1.0` (no seasonality, equivalent to `base_environment` — see
above) up to `1.5/0.5` (this module's committed default), in steps of `0.1`/`-0.1`:
`1.0/1.0`, `1.1/0.9`, `1.2/0.8`, `1.3/0.7`, `1.4/0.6`, `1.5/0.5`.

```
bash predpreygrass/non_evolutionary/base_environment_seasonal/run_season_multiplier_sweep.sh
```

Each regime is launched via `tune_ppo_base_environment_seasonal.py --season-high H --season-low
L --max-iters 500`, tagging the experiment name (`BASE_ENV_SEASONAL_HIGH<h>_LOW<l>_<timestamp>`)
so results land in distinct `~/ray_results` directories. The training script's `EpisodeReturn`
callback also logs, per episode, via `metrics_logger`: `predator_births` / `prey_births` (read
directly off the env's own `_next_predator_idx`/`_next_prey_idx` counters — the primary metric
of interest for this sweep), plus `predator_count_end` / `prey_count_end` / `grass_count_end` /
`grass_energy_mean_end` (end-of-episode population/resource snapshot).

**Status: in progress.** Results for finished regimes are moved into `~/ray_results/seasonal/`.

**Regime 1 (`1.0/1.0`) vs. regime 2 (`1.1/0.9`)**, last-50-iteration averages, single seed each
(not replicated — treat as a directional read, not a confirmed effect):

| Metric | 1.0/1.0 | 1.1/0.9 | Δ |
|---|---|---|---|
| `predator_births` | 107.6 ± 8.3 | 125.2 ± 7.0 | +16% |
| `prey_births` | 511.7 ± 16.7 | 555.0 ± 10.9 | +8% |
| `predator_count_end` | 16.5 ± 2.8 | 17.5 ± 2.4 | +6% |
| `prey_count_end` | 30.5 ± 5.2 | 24.9 ± 4.9 | -18% |
| `grass_energy_mean_end` | 0.34 ± 0.06 | 0.50 ± 0.11 | +47% |
| `episode_return_mean` | 6192.8 ± 227.9 | 6802.6 ± 152.8 | +10% |

Both regimes converge to sustained, full-length (~1001-step) episodes equally fast (by
iteration 10) — mild seasonality doesn't destabilize the ecosystem. Births, predator count, and
reward are all higher under `1.1/0.9` than the flat baseline; prey count-at-episode-end is
lower despite higher prey births, implying a higher prey death rate too. This will be updated as
regimes 3-6 (`1.2/0.8` through `1.5/0.5`) complete, to check whether this is a real
dose-response trend or single-seed noise.

<p align="center">
    <img align="center" src="../../../assets/images/gifs/rllib_pygame_1000.gif" width="600" height="500" />
</p>

### Features base environment

- At startup Predators, Prey and Grass are randomly positioned on the gridworld.

- Predators and Prey are independently (decentralized) trained via their own RLlib policy module.:

  - **Predators** (red)
  - **Prey** (blue)

- **Energy-Based Life Cycle**: Movement, hunting, and grazing consume energy—agents must act to balance survival, reproduction, and exploration.

  - Predators and Prey **learn movement strategies** based on their **partial observations**.
  - Both expend **energy** as they move around the grid and **replenish energy by eating**:

    - **Prey** eat **Grass** (green) by moving onto a grass-occupied cell.
    - **Predators** eat **Prey** by moving onto the same grid cell.

  - **Survival conditions**:

    - Both Predators and Prey must act to prevent starvation (when energy runs out).
    - Prey must act to prevent being eaten by a Predator

  - **Reproduction conditions**:

      - Both Predators and Prey reproduce **asexually** when their energy exceeds a threshold.
      - New agents are spawned near their parent.
- **Sparse rewards**: agents only receive a reward when reproducing in the base configuration. However, this can be expanded with other rewards in the [environment configuration](./config_env.py). The sparse rewards configuration is to show that the ecological system is able to sustain with this minimalistic optimized incentive for both Predators and Prey.

- Grass gradually regenerates at the same spot after being eaten by Prey. Grass, as a non-learning agent, is being regarded by the model as part of the environment, not as an actor.


## Training and evaluation results

[Training](./tune_ppo_base_environment_seasonal.py) the agents and [evaluating](./evaluate_ppo_from_checkpoint_debug.py) the environment is an example of how elaborate behaviors can emerge from simple rules in MARL models. As pointed out earlier, rewards for learning agents are solely obtained by reproduction. So all other reward options are set to zero in the environment configuration. Find more background on this [reward shaping and scaling on our website](https://humanbehaviorpatterns.org/pred-prey-grass/marl-ppg/challenges/rewards-ppg/scaling). Despite this relative sparse reward structure, maximizing these rewards results in elaborate emerging agents behaviors such as:
- Predators hunting Prey
- Multiple Predators collaborating/competing hunting Prey; increasing the probability of Prey being caught
- Prey finding and eating grass
- Predators hovering around grass to ambush Prey
- Prey trying to escape Predators


Moreover, these learning behaviors lead to more complex emergent dynamics at the ecosystem level:

- The trained policies make the ecosystem perpetuate much longer than a random policy.

- The trained agents are displaying some sort of the classic [Lotka–Volterra](https://en.wikipedia.org/wiki/Lotka%E2%80%93Volterra_equations) pattern over time:

<p align="center">
    <img src="../../../assets/images/readme/PredPreyPopulation_episode.png" width="450" height="270"/>
</p>


## Centralized versus decentralized training
The described environment and training concept is implemented with separated (decentralized) training for both learning agent types utilizing the RLlib framework. To elaborate on the difference, we compare this approach with the [(legacy) centralized trained environment utilizing PettingZoo and Stable Baselines3 (SB3)](https://github.com/doesburg11/PredPreyGrass-pettingzoo-legacy/tree/main/predpreygrass/pettingzoo).

### (Legacy) Configuration of centralized training
The MARL environment [`predpreygrass_base.py`](https://github.com/doesburg11/PredPreyGrass-pettingzoo-legacy/blob/main/predpreygrass/pettingzoo/envs/predpreygrass_base.py) is implemented using **PettingZoo**, and the agents are trained using **Stable-Baselines3 (SB3) PPO**. Essentially this solution demonstrates how SB3 can be adapted for MARL using parallel environments and centralized training. Rewards (stepping, eating, dying and reproducing) are aggregated and can be adjusted in the [environment configuration](https://github.com/doesburg11/PredPreyGrass-pettingzoo-legacy/blob/main/predpreygrass/pettingzoo/config/config_predpreygrass.py) file. Basically, Stable Baseline3 is originally designed for single-agent training. This means in this solution, training utilizes only one unified network for Predators as well Prey. See [here in more detail](https://github.com/doesburg11/PredPreyGrass-pettingzoo-legacy/tree/main/predpreygrass/pettingzoo#how-sb3-ppo-is-used-in-the-predator-prey-grass-multi-agent-setting) how SB3 PPO is used in the Predator-Prey-Grass multi-agent setting.

### Decentralized training: Pred-Prey-Grass MARL with RLlib new API stack

Obviously, using only one network has its limitations as Predators and Prey lack true specialization in their training. The RLlib new API stack framework is able to circumvent this limitation elegantly. The environment dynamics of the RLlib environments are largely the same as in the PettingZoo environment. However, newly spawned agents are placed in the vicinity of the parent, rather than randomly spawned in the entire gridworld. The implementation under-the-hood of the setup is somewhat different, utilizing array lists to store agent data rather than implementing a separate agent class (largely a result of attempting to optimize compute time of the `step` function). Similarly as in the PettingZoo environment, rewards can be adjusted in a separate environment [configuration file](./config_env.py)

Training is applied in accordance with the RLlib new API stack protocol. The training configuration is more out-of-the-box than the PettingZoo/SB3 solution, but nevertheless is much more applicable to MARL in general and especially decentralized training.

<p align="center">
    <img src="../../../assets/images/readme/multi_agent_setup.png" width="400" height="150"/>
</p>

A key difference of the decentralized training solution with the centralized training solution is that the concurrent agents become part of the environment rather than being part of a combined single "super" agent. Since, the environment of the centralized training solution consists only of static grass objects, the environment complexity of the decentralized training solution is dramatically increased. This is probably one of the reasons that training time of the RLlib solution is a multiple of the PettingZoo/SB3 solution.
