[![Python 3.11.13](https://img.shields.io/badge/python-3.11.13-blue.svg)](https://www.python.org/downloads/release/python-31113/)
[![RLlib](https://img.shields.io/badge/RLlib-v2.59.0-blue)](https://docs.ray.io/en/latest/rllib/)
[![Tests](https://github.com/doesburg11/PredPreyGrass/actions/workflows/linux-test.yml/badge.svg)](https://github.com/doesburg11/PredPreyGrass/actions/workflows/linux-test.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

# Predator-Prey-Grass
## Multi-Agent Deep Reinforcement Learning meets Darwinian and Baldwinian evolution

**A multi-agent reinforcement-learning ecosystem for studying how cooperation emerges through learning within lifetimes and evolution across generations.**

<p align="center">
    <img align="center" src="./assets/images/gifs/stag_hunt_defect.gif" width="600" height="500" />
</p>
<p align="center"><sub>Fixed-trait game-theoretic hunting: coevolution, cooperation, defection and free-riding emerging under a fixed reward design.</sub></p>

- **Darwinian evolution** of inherited traits (speed, cooperation rate, metabolic rate, and more) via reproduction and mutation
- **Baldwinian interaction** between evolution and learning tested directly — do genetic selection and learned behavior actually shape each other, or just coexist?
- **Cooperation, defection, reciprocity and coevolution as research questions**, with evidence and limits reported separately for evolving and fixed-trait populations
- Built with Python, Gymnasium and RLlib 2.59's new API stack (`RLModule` / `Learner` / `EnvRunner`), with dynamic, lifecycle-changing agent populations

**Start here:** [Quick start](#quick-start-run-a-demo-in-under-five-minutes) to run a demo in under five minutes, or [EXPERIMENTS.md](EXPERIMENTS.md) for the full catalogue of experiments.

## How it works

This project explores whether cooperative behavior, coevolution, defection, and free-riding can emerge and stabilize in a spatial, resource-limited ecosystem, by combining within-lifetime multi-agent reinforcement learning with population-level ecological and evolutionary dynamics. It probes the interplay between **nature** (inherited traits via reproduction and mutation) and **nurture** (behavior learned via reinforcement learning) — including a direct test of the **Baldwin effect**: whether genetic selection and learned behavior actually shape each other, not just coexist. Agents differ by speed, vision, energy metabolism, and decision policies — offering ground for open-ended adaptation. At its core lies a gridworld simulation where agents are not just *trained* — they are *born*, *age*, *reproduce*, *die*, and even *mutate* in a continuously changing environment.

Legacy snapshot: the pre-cleanup research codebase is archived at [PredPreyGrassLegacy](https://github.com/doesburg11/PredPreyGrassLegacy).

## Start here

The repo splits into two structurally different families of experiment, matching the
`predpreygrass/evolutionary/` vs `predpreygrass/non_evolutionary/` directory split:

- **Evolutionary**: agents carry a heritable genome trait, passed parent → offspring
  with mutation. What gets selected is discovered, not designed.
- **Non-evolutionary**: every agent trait is fixed; only the RL policy adapts. What
  emerges is a behavioral equilibrium under a given incentive design, not a change in
  the population's genetics.

The full catalogue of environments and experiments — evolutionary trials, reward-shaping
variants, cooperation/game-theory environments, and the Red Queen evaluations — lives in
**[EXPERIMENTS.md](EXPERIMENTS.md)**.

## Main findings

These are results in specific worlds and finite training/evaluation budgets. Persistent
ecology and reciprocal improvement do not by themselves establish open-ended evolution
or continuing novelty.

| Experiment | Supported finding | Main limit |
| --- | --- | --- |
| [Continuous policy co-adaptation](predpreygrass/non_evolutionary/continuous_space/base_environment_continuous/RESULTS.md#2-arms-race-in-three-runs-be-01-be-02-be-03) | In five training runs, newer predators catch fixed prey better and newer prey evade fixed predators better, including two runs with the inherited environment quirks corrected. | Shared-policy learning with fixed bodies; a finite-horizon learning arms race. |
| [Evolution plus lifetime learning](predpreygrass/evolutionary/discrete_space/eco_evolutionary_erl_baldwin/RESULTS.md#9-final-comparative-study-result-2026-08-16--erl-significantly-beats-every-other-condition) | ERL reaches the 1M-step survival cap in 83/100 runs, versus 37/100 for learning alone and 33/100 for evolution alone; p < 0.00001 against each control. | Population survival in retuned World AL; not proof of genetic assimilation or a body arms race. |
| [Prey-regulated ecology](predpreygrass/evolutionary/discrete_space/eco_evolutionary_erl_coevolution/RESULTS.md#summary) | The advantage over evolution alone persists; against learning alone it appears as prey survival in a harsh world and lower carnivore loss in an easier one. | Conditional advantage; no evolutionary arms race demonstrated. |
| [Learning steers body evolution](predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_traits/RESULTS.md) | Predator learning increases the change in predator-speed genes by +0.18 and +0.19 at two branch points (exact p = 0.005 each). | Both branch points come from CT-05, one training trajectory; shared species learning, no body arms race, mechanism partly unresolved. |
| [Visual-field evolution](predpreygrass/evolutionary/continuous_space/eco_evolutionary_continuous_fov/RESULTS.md#7-follow-up-to-test-2-freezing-at-a-late-adapted-pool-2026-10-30-to-2026-10-31) | Species' visual fields diverge, but late-adapted frozen prey coexist in 200/200 checks at each of two branches. | No evidence of body coevolution in these tests; the mirror follow-up was not completed. |
| [Cooperative deer hunting](predpreygrass/non_evolutionary/continuous_space/deer_stag_hunt/RESULTS.md#closed-2026-11-10) | Coexistence and above-chance adult approaches with a partner replicate; a payoff control is consistent with more partnering where it pays. | Partner proximity is not proof of partner seeking; modest effects, ecological confounds and unexplained late declines. |
| [Reward shaping](predpreygrass/non_evolutionary/discrete_space/project_reward_shaping/README.md#8-conclusion) | Sparse reproduction-only rewards outperform the four tested shaping variants on reproduction rates. | One training run per condition; proposed mechanisms remain hypotheses. |
| [Discrete Red Queen](predpreygrass/non_evolutionary/discrete_space/red_queen/README.md) | A multi-seed, multi-checkpoint evaluation harness is implemented. | The original single-seed result was retracted; no replacement result is recorded. |

The [evolutionary deer-hunting extension](predpreygrass/evolutionary/continuous_space/eco_evolutionary_deer_stag_hunt/RESULTS.md)
has no final comparative verdict recorded: two pilot designs failed, a third passed,
and the latest record documents interrupted/restarted main runs. Its proposed division
between innate attraction and learned hunting remains exploratory.

## Quick start: run a demo in under five minutes

```bash
git clone https://github.com/doesburg11/PredPreyGrass.git
cd PredPreyGrass
python -m venv .venv && source .venv/bin/activate
pip install -e .
python ./predpreygrass/non_evolutionary/discrete_space/base_environment_discrete/random_policy.py
```

This installs the base dependencies and runs a random policy in the base environment —
no VS Code or Conda required. `pygame` (for the rendered window) ships as a regular pip
dependency; the Conda/GCC setup below is only needed if your platform lacks a
prebuilt `pygame` wheel.

Pretrained checkpoints and historical training outputs are preserved in the legacy archive rather than shipped in the active source tree.

## Full setup (Visual Studio Code + Conda)

**Editor used:** Visual Studio Code on Linux Mint 22.0 Cinnamon

1. Clone the repository:
   ```bash
   git clone https://github.com/doesburg11/PredPreyGrass.git
   ```
2. Open Visual Studio Code and execute:
   - Press `ctrl+shift+p`
   - Type and choose: "Python: Create Environment..."
   - Choose environment: Conda
   - Choose interpreter: Python 3.11.13 or higher
   - Open a new terminal
   - ```bash
     pip install -e .
     ```
3. Install the additional system dependency for Pygame visualization (only needed if
   `pip install` can't find a prebuilt `pygame` wheel for your platform):
    -   ```bash
        conda install -y -c conda-forge gcc=14.2.0
        ```

## Acknowledgments

Developed with AI coding assistance from [Claude](https://claude.com/claude-code) (Anthropic), which does the implementation, with [Codex](https://openai.com/codex) (OpenAI) acting as an independent second opinion, peer-reviewing Claude's nontrivial code changes.

## References

- [RLlib: Industry-Grade, Scalable Reinforcement Learning](https://docs.ray.io/en/master/rllib/index.html)
- [Paper Collection of Multi-Agent Reinforcement Learning (MARL)](https://github.com/LantaoYu/MARL-Papers)
- [Multi-Agent Reinforcement Learning: Foundations and Modern Approaches. Stefano V. Albrecht, Filippos Christianos, and Lukas Schäfer](https://www.marl-book.com/download/)

## Citation

If you use this software in your research, please cite it — see [CITATION.cff](CITATION.cff).
