"""Standalone wall placement sanity check without requiring full Ray tune stack.

Run:
    python -m predpreygrass.rllib.walls_occlusion.wall_sanity_check --grid 10 --walls 20 \
        --pred 2 --prey 3 --grass 5 --seed 123

Outputs wall count, sample coordinates, and verifies no overlap with agents or grass.
"""
from __future__ import annotations
import argparse
from predpreygrass.rllib.walls_occlusion.predpreygrass_rllib_env import PredPreyGrass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=10)
    ap.add_argument("--walls", type=int, default=20)
    ap.add_argument("--pred", type=int, default=2)
    ap.add_argument("--prey", type=int, default=3)
    ap.add_argument("--grass", type=int, default=5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    cfg = dict(
        grid_size=args.grid,
        num_walls=args.walls,
        n_initial_active_type_1_predator=args.pred,
        n_initial_active_type_1_prey=args.prey,
        initial_num_grass=args.grass,
        num_obs_channels=4,
        predator_obs_range=7,
        prey_obs_range=5,
    )
    env = PredPreyGrass(cfg)
    obs, _ = env.reset(seed=args.seed)
    walls = env.wall_positions
    print(f"Walls count: {len(walls)} (expected {args.walls})")
    print("First few walls:", list(walls)[:8])
    overlap_agents = any(pos in walls for pos in env.agent_positions.values())
    overlap_grass = any(pos in walls for pos in env.grass_positions.values())
    print("Agent-wall overlap:", overlap_agents)
    print("Grass-wall overlap:", overlap_grass)
    any_obs = next(iter(obs.values()))
    print("Sample obs shape:", any_obs.shape)


if __name__ == "__main__":  # pragma: no cover
    main()
