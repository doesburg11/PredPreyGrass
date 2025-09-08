from __future__ import annotations

"""
Print five consecutive steps for type_1_predator_0:
- Observation prey channel (ch2) 7x7
- DFT prey-trace u for the same step

Run:
  PYTHONPATH=./src /home/doesburg/Projects/PredPreyGrass/.conda/bin/python -m predpreygrass.rllib.dynamic_field_theory.debug.print_preych_u_five_steps
"""

import numpy as np
import torch

from predpreygrass.rllib.dynamic_field_theory.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.dynamic_field_theory.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.dynamic_field_theory.models.dft_memory_model import DFTMemoryConvModel


def fmt(mat: np.ndarray, prec: int = 2) -> str:
    return "\n".join(" ".join(f"{v:.{prec}f}" for v in row) for row in mat)


def main():
    cfg = dict(config_env)
    env = PredPreyGrass(cfg)
    obs, _ = env.reset(seed=cfg.get("seed", 42))

    target = "type_1_predator_0"
    if target not in obs:
        print(f"{target} not active at reset; aborting.")
        return

    # Build DFT model for a single agent obs, only prey trace (dual_traces=False)
    obs_space = env.observation_spaces[target]
    act_space = env.action_spaces[target]
    model = DFTMemoryConvModel(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=act_space.n,
        model_config={
            "custom_model_config": {
                "prey_channel_index": 2,
                "dual_traces": False,
                "dft_dt": 1.0,
                "dft_tau": 10.0,
                "dft_gamma": 0.55,
                "dft_exc_sigma": 1.2,
                "dft_exc_gain": 0.60,
                "dft_input_gain": 1.60,
                "dft_zmin": 0.0,
                "dft_zmax": 1.0,
                "dft_kernel_size": 5,
            }
        },
        name="debug_dft_u_only",
    )

    state = model.get_initial_state()

    # Step forward until prey is visible in FOV of target
    def prey_visible(ob):
        ch2 = ob[2]
        return np.any(ch2 > 0)

    steps_wait = 0
    while not prey_visible(obs[target]) and steps_wait < 50:
        actions = {}
        for aid in env.agents:
            if "predator" in aid:
                actions[aid] = 4  # stay
            else:
                # random legal move
                if "type_1" in aid:
                    actions[aid] = np.random.randint(0, env.type_1_act_range ** 2)
                else:
                    actions[aid] = np.random.randint(0, env.type_2_act_range ** 2)
        obs, _, terms, truncs, _ = env.step(actions)
        if target not in obs:
            print(f"{target} not present at step {env.current_step}; aborting.")
            return
        steps_wait += 1

    if not prey_visible(obs[target]):
        print("Prey not visible within 50 steps; aborting.")
        return

    # Print five consecutive steps
    for i in range(5):
        ob = obs[target]
        prey_ch = ob[2]
        # Forward pass to update u for this obs
        ob_t = torch.tensor(ob, dtype=torch.float32).unsqueeze(0)
        logits, state_out = model({"obs": ob_t}, state, None)
        u = state_out[0][0].cpu().numpy()  # [H,W]

        print(f"Step {env.current_step} — {target}")
        print("Observation prey channel (ch2):")
        print(fmt(prey_ch))
        print("DFT prey-trace u:")
        print(fmt(u))
        print()

        # Feed state forward
        state = state_out

        # Advance env one step using simple actions
        actions = {}
        for aid in env.agents:
            if "predator" in aid:
                actions[aid] = 4
            else:
                if "type_1" in aid:
                    actions[aid] = np.random.randint(0, env.type_1_act_range ** 2)
                else:
                    actions[aid] = np.random.randint(0, env.type_2_act_range ** 2)
        obs, _, terms, truncs, _ = env.step(actions)
        if target not in obs:
            print(f"{target} disappeared at step {env.current_step}; stopping.")
            break


if __name__ == "__main__":
    main()
