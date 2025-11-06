"""
Vectorized movement utilities for the walls_occlusion_simplified environment.

This module provides an optional, drop-in helper to compute agent movements and
energy updates in a vectorized fashion when line-of-sight movement constraints
are disabled (respect_los_for_movement == False).

It does NOT modify the environment; you can import and call from your env
to replace only the movement hot-path while keeping semantics identical for
the LOS-disabled case.

Usage (example):

    from predpreygrass.rllib.walls_occlusion_simplified.utils.vectorized_movement import (
        process_agent_movements_vectorized_no_los,
    )
    def _process_agent_movements(self, action_dict):
        """
        Process movement, energy cost, and grid updates for all agents.
        """
        # Vectorized movement for all agents in action_dict
        agent_ids = [agent for agent in action_dict if agent in self.agent_positions]
        old_positions = np.array([self.agent_positions[agent] for agent in agent_ids])
        actions = np.array([action_dict[agent] for agent in agent_ids])
        new_positions = []
        for i, agent in enumerate(agent_ids):
            new_pos = self._get_move(agent, actions[i])
            new_positions.append(new_pos)
        new_positions = np.array(new_positions)

        # Vectorized grid state update for all agents
        move_costs = np.array([
            self._get_movement_energy_cost(agent, tuple(old_positions[i]), tuple(new_positions[i]))
            for i, agent in enumerate(agent_ids)
        ])
        for i, agent in enumerate(agent_ids):
            self.agent_energies[agent] -= move_costs[i]
            self._per_agent_step_deltas[agent]["move"] = -move_costs[i]
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["distance_traveled"] += np.linalg.norm(np.array(new_positions[i]) - np.array(old_positions[i]))
            self.unique_agent_st    def _process_agent_movements(self, action_dict):
        """
        Process movement, energy cost, and grid updates for all agents.
        """
        # Vectorized movement for all agents in action_dict
        agent_ids = [agent for agent in action_dict if agent in self.agent_positions]
        old_positions = np.array([self.agent_positions[agent] for agent in agent_ids])
        actions = np.array([action_dict[agent] for agent in agent_ids])
        new_positions = []
        for i, agent in enumerate(agent_ids):
            new_pos = self._get_move(agent, actions[i])
            new_positions.append(new_pos)
        new_positions = np.array(new_positions)

        # Vectorized grid state update for all agents
        move_costs = np.array([
            self._get_movement_energy_cost(agent, tuple(old_positions[i]), tuple(new_positions[i]))
            for i, agent in enumerate(agent_ids)
        ])
        for i, agent in enumerate(agent_ids):
            self.agent_energies[agent] -= move_costs[i]
            self._per_agent_step_deltas[agent]["move"] = -move_costs[i]
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["distance_traveled"] += np.linalg.norm(np.array(new_positions[i]) - np.array(old_positions[i]))
            self.unique_agent_stats[uid]["energy_spent"] += move_costs[i]
            self.unique_agent_stats[uid]["avg_energy_sum"] += self.agent_energies[agent]
            self.unique_agent_stats[uid]["avg_energy_steps"] += 1
            self.agent_positions[agent] = tuple(new_positions[i])
        # Prepare masks for predators and prey
        is_predator = np.array(["predator" in agent for agent in agent_ids])
        is_prey = np.array(["prey" in agent for agent in agent_ids])
        # Old and new positions for each type
        old_pred = old_positions[is_predator]
        new_pred = new_positions[is_predator]
        pred_energies = np.array([self.agent_energies[agent_ids[i]] for i in range(len(agent_ids)) if is_predator[i]])
        old_prey = old_positions[is_prey]
        new_prey = new_positions[is_prey]
        prey_energies = np.array([self.agent_energies[agent_ids[i]] for i in range(len(agent_ids)) if is_prey[i]])
        # Zero out old positions
        if len(old_pred) > 0:
            self.grid_world_state[1, old_pred[:,0], old_pred[:,1]] = 0
        if len(old_prey) > 0:
            self.grid_world_state[2, old_prey[:,0], old_prey[:,1]] = 0
        # Set new positions with updated energies
        if len(new_pred) > 0:
            self.grid_world_state[1, new_pred[:,0], new_pred[:,1]] = pred_energies
        if len(new_prey) > 0:
            self.grid_world_state[2, new_prey[:,0], new_prey[:,1]] = prey_energies
        # Logging (optional, keep per-agent for now)
        for i, agent in enumerate(agent_ids):
            old_position = tuple(old_positions[i])
            new_position = tuple(new_positions[i])
            move_cost = move_costs[i]
            self._log(
                self.verbose_movement,
                f"[MOVE] {agent} moved: {tuple(map(int, old_position))} -> {tuple(map(int, new_position))}. "
                f"Move energy: {move_cost:.2f} Energy level: {self.agent_energies[agent]:.2f}\n",
                "blue",
            )
            self.unique_agent_stats[uid]["avg_energy_sum"] += self.agent_energies[agent]
            self.unique_agent_stats[uid]["avg_energy_steps"] += 1
            self.agent_positions[agent] = tuple(new_positions[i])
        # Prepare masks for predators and prey
        is_predator = np.array(["predator" in agent for agent in agent_ids])
        is_prey = np.array(["prey" in agent for agent in agent_ids])
        # Old and new positions for each type
        old_pred = old_positions[is_predator]
        new_pred = new_positions[is_predator]
        pred_energies = np.array([self.agent_energies[agent_ids[i]] for i in range(len(agent_ids)) if is_predator[i]])
        old_prey = old_positions[is_prey]
        new_prey = new_positions[is_prey]
        prey_energies = np.array([self.agent_energies[agent_ids[i]] for i in range(len(agent_ids)) if is_prey[i]])
        # Zero out old positions
        if len(old_pred) > 0:
            self.grid_world_state[1, old_pred[:,0], old_pred[:,1]] = 0
        if len(old_prey) > 0:
            self.grid_world_state[2, old_prey[:,0], old_prey[:,1]] = 0
        # Set new positions with updated energies
        if len(new_pred) > 0:
            self.grid_world_state[1, new_pred[:,0], new_pred[:,1]] = pred_energies
        if len(new_prey) > 0:
            self.grid_world_state[2, new_prey[:,0], new_prey[:,1]] = prey_energies
        # Logging (optional, keep per-agent for now)
        for i, agent in enumerate(agent_ids):
            old_position = tuple(old_positions[i])
            new_position = tuple(new_positions[i])
            move_cost = move_costs[i]
            self._log(
                self.verbose_movement,
                f"[MOVE] {agent} moved: {tuple(map(int, old_position))} -> {tuple(map(int, new_position))}. "
                f"Move energy: {move_cost:.2f} Energy level: {self.agent_energies[agent]:.2f}\n",
                "blue",
            )
    # inside your env._process_agent_movements(self, action_dict):
    if not self.respect_los_for_movement:
        process_agent_movements_vectorized_no_los(self, action_dict)
        return
    # else, fall back to the existing per-agent path (keeps LOS logic)

The function here performs the following steps:
  - Collect acting agent ids, old positions, and actions as numpy arrays
  - Compute proposed new positions using prebuilt action maps (type_1 vs type_2)
  - Clip to bounds and block moves into walls or same-type occupied cells
  - Compute movement energy costs vectorized and update per-agent stats
  - Write back energies, positions, and grid updates for predator/prey layers

Notes:
  - This preserves the current semantics for the case where LOS constraints are OFF.
  - When LOS constraints are ON, keep using your existing per-agent method
    (_get_move) because LOS/corner-cut checks are scalar/branchy.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


def _build_action_map_np(range_size: int) -> np.ndarray:
    """
    Build a numpy action map of shape (range_size^2, 2) with (dx, dy) pairs,
    using the same enumeration order as the env's _generate_action_map:
      for dx in [-d..d]:
        for dy in [-d..d]:
          yield (dx, dy)
    where d = (range_size - 1) // 2.
    """
    delta = (range_size - 1) // 2
    moves = [(dx, dy) for dx in range(-delta, delta + 1) for dy in range(-delta, delta + 1)]
    return np.asarray(moves, dtype=np.int32)


def _ensure_action_maps_cached(env) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build and cache numpy action maps on the env instance for reuse.
    Returns (map_type1, map_type2), each of shape (range^2, 2).
    """
    m1 = getattr(env, "_action_map_np_type_1", None)
    m2 = getattr(env, "_action_map_np_type_2", None)
    if m1 is None:
        m1 = _build_action_map_np(env.type_1_act_range)
        setattr(env, "_action_map_np_type_1", m1)
    if m2 is None:
        m2 = _build_action_map_np(env.type_2_act_range)
        setattr(env, "_action_map_np_type_2", m2)
    return m1, m2


def process_agent_movements_vectorized_no_los(env, action_dict: Dict[str, int]) -> None:
    """
    Vectorized movement+energy+grid update for all agents in action_dict,
    for the case when env.respect_los_for_movement is False.

    This directly mutates env state (positions, energies, per-step deltas,
    unique stats, and grid layers 1/2), matching the semantics of the
    existing per-agent implementation without LOS checks.
    """
    if getattr(env, "respect_los_for_movement", True):
        raise RuntimeError(
            "process_agent_movements_vectorized_no_los called while respect_los_for_movement=True"
        )

    # Collect acting agents that are still alive/present
    agent_ids = [aid for aid in action_dict if aid in env.agent_positions]
    if not agent_ids:
        return

    # Arrays of positions and actions
    old_positions = np.asarray([env.agent_positions[a] for a in agent_ids], dtype=np.int32)
    actions = np.asarray([int(action_dict[a]) for a in agent_ids], dtype=np.int32)

    # Build action maps and per-agent move vectors based on type_1/type_2
    map1, map2 = _ensure_action_maps_cached(env)
    is_type1 = np.fromiter(("type_1" in a for a in agent_ids), count=len(agent_ids), dtype=bool)
    moves_type1 = map1[actions]
    moves_type2 = map2[actions]
    moves = np.where(is_type1[:, None], moves_type1, moves_type2)  # (N,2)

    # Propose new positions and clip to bounds
    new_positions = old_positions + moves
    np.clip(new_positions, 0, env.grid_size - 1, out=new_positions)

    # Block entries into walls (channel 0 is walls as 1.0)
    walls_mask = env.grid_world_state[0]  # (G, G), float32 with 1.0 where wall
    blocked_by_wall = walls_mask[new_positions[:, 0], new_positions[:, 1]] > 0.0

    # Block entries into already-occupied same-type cells using pre-move occupancy
    is_predator = np.fromiter(("predator" in a for a in agent_ids), count=len(agent_ids), dtype=bool)
    is_prey = ~is_predator
    occ_pred = env.grid_world_state[1]  # (G, G)
    occ_prey = env.grid_world_state[2]  # (G, G)
    occ_pred_hits = occ_pred[new_positions[:, 0], new_positions[:, 1]] > 0.0
    occ_prey_hits = occ_prey[new_positions[:, 0], new_positions[:, 1]] > 0.0
    blocked_by_occ = np.where(is_predator, occ_pred_hits, occ_prey_hits)

    # Combine block reasons and revert to old position where blocked
    blocked = blocked_by_wall | blocked_by_occ
    new_positions[blocked] = old_positions[blocked]

    # Movement costs: distance * factor * current_energy
    dx = new_positions[:, 0] - old_positions[:, 0]
    dy = new_positions[:, 1] - old_positions[:, 1]
    distances = np.sqrt(dx.astype(np.float32) ** 2 + dy.astype(np.float32) ** 2)
    move_factor = float(env.config["move_energy_cost_factor"])  # scalar
    energies = np.asarray([env.agent_energies[a] for a in agent_ids], dtype=np.float32)
    move_costs = distances * move_factor * energies

    # Apply energy updates and per-agent stats; write positions back
    # Update energies first to reuse updated values below
    new_energies = energies - move_costs
    for i, aid in enumerate(agent_ids):
        env.agent_energies[aid] = float(new_energies[i])
        env._per_agent_step_deltas[aid]["move"] = float(-move_costs[i])
        uid = env.unique_agents[aid]
        # distance traveled (Euclidean)
        env.unique_agent_stats[uid]["distance_traveled"] += float(
            np.linalg.norm(new_positions[i].astype(np.float32) - old_positions[i].astype(np.float32))
        )
        env.unique_agent_stats[uid]["energy_spent"] += float(move_costs[i])
        env.unique_agent_stats[uid]["avg_energy_sum"] += env.agent_energies[aid]
        env.unique_agent_stats[uid]["avg_energy_steps"] += 1
        env.agent_positions[aid] = (int(new_positions[i, 0]), int(new_positions[i, 1]))

    # Grid updates: zero out old positions per type, then set new positions with updated energies
    old_pred = old_positions[is_predator]
    new_pred = new_positions[is_predator]
    pred_energies = new_energies[is_predator]
    old_prey = old_positions[is_prey]
    new_prey = new_positions[is_prey]
    prey_energies = new_energies[is_prey]

    if old_pred.size:
        env.grid_world_state[1, old_pred[:, 0], old_pred[:, 1]] = 0.0
    if old_prey.size:
        env.grid_world_state[2, old_prey[:, 0], old_prey[:, 1]] = 0.0
    if new_pred.size:
        env.grid_world_state[1, new_pred[:, 0], new_pred[:, 1]] = pred_energies
    if new_prey.size:
        env.grid_world_state[2, new_prey[:, 0], new_prey[:, 1]] = prey_energies

    # Optional per-agent movement logs (kept for parity with current behavior)
    if getattr(env, "verbose_movement", False):
        for i, aid in enumerate(agent_ids):
            old_pos = (int(old_positions[i, 0]), int(old_positions[i, 1]))
            new_pos = (int(new_positions[i, 0]), int(new_positions[i, 1]))
            env._log(
                env.verbose_movement,
                f"[MOVE] {aid} moved: {old_pos} -> {new_pos}. Move energy: {move_costs[i]:.2f} "
                f"Energy level: {env.agent_energies[aid]:.2f}\n",
                "blue",
            )
