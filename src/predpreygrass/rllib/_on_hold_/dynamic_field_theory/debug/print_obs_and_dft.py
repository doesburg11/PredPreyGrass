from __future__ import annotations

"""
Print a real per-step example of a predator seeing prey:
- Raw observation prey channel (ch2) as a 7x7 matrix.
- DFT memory channels (u: prey-trace, v: predator-trace) produced for the same obs.

Run as a module (ensure PYTHONPATH includes ./src):
    python -m predpreygrass.rllib.dynamic_field_theory.debug.print_obs_and_dft
"""

import os
from typing import Tuple

import numpy as np
import torch

from predpreygrass.rllib.dynamic_field_theory.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.dynamic_field_theory.config.config_env_train_v1_0 import config_env
from predpreygrass.rllib.dynamic_field_theory.models.dft_memory_model import DFTMemoryConvModel


def fmt_mat(mat: np.ndarray, prec: int = 2) -> str:
    rows = []
    for r in mat:
        rows.append(" ".join(f"{v:.{prec}f}" for v in r))
    return "\n".join(rows)


def find_predator_with_prey_in_fov(obs: dict) -> Tuple[str, np.ndarray]:
    for agent_id, ob in obs.items():
        if "predator" not in agent_id:
            continue
        prey_ch = ob[2]  # channel 2 is prey
        if np.any(prey_ch > 0):
            return agent_id, ob
    return None, None


def main():
    # Use a fixed seed for reproducibility
    config = dict(config_env)
    seed = config.get("seed", 42)
    env = PredPreyGrass(config)
    obs, _ = env.reset(seed=seed)

    agent_id, agent_obs = find_predator_with_prey_in_fov(obs)

    # Step a few times if initially none in FOV
    steps = 0
    while agent_id is None and steps < 30:
        actions = {}
        for aid in env.agents:
            # center (no move) for predators in 3x3 map → index 4
            if "predator" in aid:
                actions[aid] = 4
            else:
                # random legal action for prey (type_1 range by default)
                if "type_1" in aid:
                    actions[aid] = np.random.randint(0, env.type_1_act_range ** 2)
                else:
                    actions[aid] = np.random.randint(0, env.type_2_act_range ** 2)
        obs, _, terms, truncs, _ = env.step(actions)
        agent_id, agent_obs = find_predator_with_prey_in_fov(obs)
        steps += 1

    if agent_id is None:
        print("No predator with prey in FOV found within 30 steps. Try re-running.")
        return

    # Extract the prey channel (without DFT)
    prey_ch = agent_obs[2]  # (7x7) for predators

    # Prepare a DFT model to compute memory for this single observation
    obs_space = env.observation_spaces[agent_id]
    act_space = env.action_spaces[agent_id]

    model = DFTMemoryConvModel(
        obs_space=obs_space,
        action_space=act_space,
        num_outputs=act_space.n,
        model_config={
            "custom_model_config": {
                "prey_channel_index": 2,
                "predator_channel_index": 1,
                "dual_traces": True,
                # DFT params (same defaults as training script)
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
        name="debug_dft",
    )

    # Single forward pass with zero initial memory
    obs_tensor = torch.tensor(agent_obs, dtype=torch.float32).unsqueeze(0)
    state_in = model.get_initial_state()
    logits, state_out = model({"obs": obs_tensor}, state_in, None)

    # state_out holds [u_next, v_next] when dual_traces=True, each shape [B,H,W]
    u_next = state_out[0][0].cpu().numpy()
    v_next = state_out[1][0].cpu().numpy() if len(state_out) > 1 else None

    # Print matrices
    print(f"Step={env.current_step}, Agent={agent_id}")
    print("\nObservation prey channel (ch2) 7x7:")
    print(fmt_mat(prey_ch, prec=2))

    print("\nDFT prey-trace u (added as extra channel):")
    print(fmt_mat(u_next, prec=2))

    if v_next is not None:
        print("\nDFT predator-trace v (second extra channel):")
        print(fmt_mat(v_next, prec=2))


if __name__ == "__main__":
    main()
