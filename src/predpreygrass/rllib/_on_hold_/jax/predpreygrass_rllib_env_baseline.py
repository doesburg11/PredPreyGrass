import numpy as np
from .predpreygrass_rllib_env import PredPreyGrass


class PredPreyGrassBaseline(PredPreyGrass):
    """Baseline (pre-Step1) behavior:
    - Linear scans for predator->prey and prey->grass engagements
    - Always sorts agents list each step
    Used only for benchmarking relative speed gain.
    """

    def _handle_predator_engagement(self, agent, observations, rewards, terminations, truncations):  # type: ignore[override]
        predator_position = self.agent_positions[agent]
        caught_prey = next(
            (prey for prey, pos in self.agent_positions.items() if "prey" in prey and np.array_equal(predator_position, pos)),
            None,
        )
        if caught_prey:
            # delegate to parent style by duplicating simplified logic (no index removal)
            self._log(
                self.verbose_engagement,
                f"[ENGAGE] {agent} caught {caught_prey} at {tuple(map(int, predator_position))}",
                "white",
            )
            self.agents_just_ate.add(agent)
            rewards[agent] = self._get_type_specific("reward_predator_catch_prey", agent)
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]
            raw_gain = min(self.agent_energies[caught_prey], self.config.get("max_energy_gain_per_prey", float("inf")))
            efficiency = self.config.get("energy_transfer_efficiency", 1.0)
            gain = raw_gain * efficiency
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain
            max_energy = self.config.get("max_energy_predator", float("inf"))
            self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["times_ate"] += 1
            self.unique_agent_stats[uid]["energy_gained"] += self.agent_energies[caught_prey]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            self.grid_world_state[1, *predator_position] = self.agent_energies[agent]
            observations[caught_prey] = self._get_observation(caught_prey)
            rewards[caught_prey] = self._get_type_specific("penalty_prey_caught", caught_prey)
            self.cumulative_rewards.setdefault(caught_prey, 0.0)
            self.cumulative_rewards[caught_prey] += rewards[caught_prey]
            terminations[caught_prey] = True
            truncations[caught_prey] = False
            self.active_num_prey -= 1
            self.grid_world_state[2, *self.agent_positions[caught_prey]] = 0
            uidp = self.unique_agents[caught_prey]
            stat = self.unique_agent_stats[uidp]
            stat["death_step"] = self.current_step
            stat["death_cause"] = "eaten"
            stat["final_energy"] = self.agent_energies[agent]
            steps = max(stat["avg_energy_steps"], 1)
            stat["avg_energy"] = stat["avg_energy_sum"] / steps
            stat["cumulative_reward"] = self.cumulative_rewards.get(agent, 0.0)
            self.death_agents_stats[uidp] = stat
            del self.agent_positions[caught_prey]
            del self.prey_positions[caught_prey]
            del self.agent_energies[caught_prey]
        else:
            rewards[agent] = self._get_type_specific("reward_predator_step", agent)
        observations[agent] = self._get_observation(agent)
        self.cumulative_rewards.setdefault(agent, 0)
        self.cumulative_rewards[agent] += rewards[agent]
        terminations[agent] = False
        truncations[agent] = False

    def _handle_prey_engagement(self, agent, observations, rewards, terminations, truncations):  # type: ignore[override]
        if terminations.get(agent):
            return
        prey_position = self.agent_positions[agent]
        caught_grass = next(
            (g for g, pos in self.grass_positions.items() if "grass" in g and np.array_equal(prey_position, pos)),
            None,
        )
        if caught_grass:
            self._log(
                self.verbose_engagement,
                f"[ENGAGE] {agent} caught grass at {tuple(map(int, prey_position))}",
                "white",
            )
            self.agents_just_ate.add(agent)
            rewards[agent] = self._get_type_specific("reward_prey_eat_grass", agent)
            self.cumulative_rewards.setdefault(agent, 0)
            self.cumulative_rewards[agent] += rewards[agent]
            raw_gain = min(self.grass_energies[caught_grass], self.config.get("max_energy_gain_per_grass", float("inf")))
            efficiency = self.config.get("energy_transfer_efficiency", 1.0)
            gain = raw_gain * efficiency
            self.agent_energies[agent] += gain
            self._per_agent_step_deltas[agent]["eat"] = gain
            max_energy = self.config.get("max_energy_prey", float("inf"))
            self.agent_energies[agent] = min(self.agent_energies[agent], max_energy)
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["times_ate"] += 1
            self.unique_agent_stats[uid]["energy_gained"] += self.grass_energies[caught_grass]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
            self.grid_world_state[2, *prey_position] = self.agent_energies[agent]
            self.grid_world_state[3, *prey_position] = 0
            self.grass_energies[caught_grass] = 0
        else:
            rewards[agent] = self._get_type_specific("reward_prey_step", agent)
            uid = self.unique_agents[agent]
            self.unique_agent_stats[uid]["cumulative_reward"] += rewards[agent]
        observations[agent] = self._get_observation(agent)
        self.cumulative_rewards.setdefault(agent, 0)
        self.cumulative_rewards[agent] += rewards[agent]
        terminations[agent] = False
        truncations[agent] = False

    def step(self, action_dict):  # type: ignore[override]
        obs, rew, terms, truncs, infos = super().step(action_dict)
        # Force per-step sorting (baseline behavior)
        self.agents.sort()
        return obs, rew, terms, truncs, infos
