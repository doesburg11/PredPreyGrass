"""
Prey-density-floor intervention for the hunting-success necessity question (Codex's suggestion; see
RESULTS.md, Iteration 11). Whenever the prey population drops below `prey_density_floor` after a
step's deaths/hunts/reproduction are resolved, immediately spawn enough replacement prey (default
initial energy, at an empty cell, using the same spawn-position helper and agent-ID pool as normal
prey reproduction) to bring it back to the floor. Called "matched ecology" loosely in early notes;
"fixed prey-density" is the more accurate name (a third-party review pointed this out): only the
prey COUNT is held near a floor. Predator abundance, total catch throughput (which grew to about
1,900 catches per episode in the SUCCESS_ONLY_MATCHED run), and the energy injected into the system
by each replacement (a full initial prey's worth) are not held fixed at all, and are known to differ
substantially between runs -- so this is a partial control, not a fully matched environment.

Purpose: the ABL_SUCCESS_ONLY / ABL_EQUAL_ODDS ablations (predator_sexual_reproduction/RESULTS.md,
Iteration 8) removed the k=0.5 division of labor, but also collapsed the prey population (2.9, 0.4,
0.5 prey alive at the end for ABL_SUCCESS_ONLY's three seeds) -- so it is unclear whether raising
women's hunting success removes the split, or whether the collapsing ecology (short, prey-poor
lives; population feedback) is doing the work instead. This class removes that confound by holding
prey density above a floor, so a high-success run can be compared on a healthy-ish ecology.

Deliberately narrower than the abandoned prey/grass SUPPLY boost (n_initial_active_prey,
initial_num_grass, energy_gain_per_step_grass -- see tune_ppo_predator_sexual_reproduction.py and
RESULTS.md's "matched-ecology attempt, abandoned"): boosting the STARTING supply gives predators an
energy windfall (more prey to hunt from step 0, on top of unchanged fruit), which triggered a
predator population boom that recrashed the boosted prey pool just as fast or faster (two
calibration runs, 3x and 5x boost, both collapsed to 0 prey within 10 iterations, the bigger boost
producing the bigger predator boom: ~55 vs ~93 total predators alive by iteration 10). Replacing
losses one at a time, instead of front-loading supply, avoids that windfall: predators still face
the same per-capita energy economy as in every other run in this module; only the prey floor is
enforced, by topping up after the fact rather than by increasing what predators start with.

Reproduction, energy costs, and hunting odds are not modified by this class; this only adds
replacement spawns after the base class's Step 5 (reproduction) has run, using the same
`_find_available_spawn_position` helper and `_next_prey_idx` ID pool as normal prey reproduction
(predpreygrass_rllib_env.py, Step 5a). A replacement spawn is NOT a birth: it pays no reward, and
`episode_births["prey"]` is not incremented, so `births_prey` in the TensorBoard metrics still
reflects only real (energy-threshold-triggered) reproduction. `current_num_prey` and
`final_num_prey` DO include replacements, since those track the actual population the environment
presents to the agents. Each replacement is still an ENERGY intervention, though (see above): it
injects a full initial prey's worth of energy into the system, on top of whatever normal
reproduction and grass regrowth already provide.

KNOWN LIMITATION (see RESULTS.md, Iteration 11): under heavy enough hunting pressure, the
per-episode `n_possible_prey` ID budget (shared with normal births) can itself be exhausted within a
single long episode, after which replenishment silently stops and prey collapse resumes. This is a
design-breaking limitation, not merely an inconvenience -- the environment stops implementing its
defining intervention once exhausted, with no error or warning by default (only a print if
`verbose_spawning`). `prey_density_floor` is validated only for being non-negative; a floor that
exceeds the ID pool or the grid's spatial capacity is not detected and silently becomes
unenforceable. Not fixed here; a larger pool, batched replenishment, or capping predator
reproduction are the candidate fixes.

Predator extinction (either sex reaching 0) still ends the episode normally -- this class narrowly
targets the prey-collapse confound, not predator population dynamics, which the earlier equal-odds/
success-only ablations did not show any problem with (their predator populations stayed positive,
just small). Max-steps truncation is also untouched.
"""
from predpreygrass.non_evolutionary.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass


class FixedPreyDensityEnv(PredPreyGrass):
    def __init__(self, config):
        super().__init__(config)
        self.prey_density_floor = int(config.get("prey_density_floor", 20))
        if self.prey_density_floor < 0:
            raise ValueError(f"prey_density_floor must be >= 0 (got {self.prey_density_floor})")

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = super().step(action_dict)

        # Max-steps truncation (the base class's early-return branch): leave untouched.
        if truncations.get("__all__", False):
            return observations, rewards, terminations, truncations, infos

        # Predator extinction ends the episode normally; this class only targets prey collapse.
        if self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0:
            return observations, rewards, terminations, truncations, infos

        replenished = False
        if self.current_num_prey < self.prey_density_floor:
            replenished = self._replenish_prey_to_floor(rewards, terminations, truncations)

        if replenished:
            # Codex review caught: the base class already generated final observations before this
            # override runs (predpreygrass_rllib_env.py's Step 6), so without this, every OTHER live
            # agent's returned observation would be stale -- it wouldn't show the newly spawned prey.
            # Regenerate for every live agent, exactly matching the base class's Step 6 pattern.
            for agent in self.agents:
                if agent in self.agent_positions:
                    observations[agent] = self._get_observation(agent)

        # Re-derive using the same formula the base class uses (predpreygrass_rllib_env.py's Step 6):
        # correct in every case, since replenishment can only raise current_num_prey, never lower it.
        terminations["__all__"] = (
            self.current_num_prey <= 0 or self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0
        )
        return observations, rewards, terminations, truncations, infos

    def _replenish_prey_to_floor(self, rewards, terminations, truncations):
        """Returns True iff at least one replacement was actually spawned (the caller uses this to
        decide whether every agent's observation needs to be regenerated)."""
        deficit = self.prey_density_floor - self.current_num_prey
        occupied_positions = set(self.agent_positions.values())
        spawned_any = False
        for _ in range(deficit):
            if self._next_prey_idx >= self.n_possible_prey:
                if self.verbose_spawning:
                    print("FixedPreyDensityEnv: no new prey agent IDs left in the pool this episode")
                break
            reference_position = (
                int(self.rng.integers(self.grid_size)),
                int(self.rng.integers(self.grid_size)),
            )
            new_position = self._find_available_spawn_position(reference_position, occupied_positions)
            if new_position is None:
                if self.verbose_spawning:
                    print("FixedPreyDensityEnv: no free spawn position available for a replacement prey")
                break

            new_agent = f"prey_{self._next_prey_idx}"
            self._next_prey_idx += 1
            self.agents.append(new_agent)
            self.agent_positions[new_agent] = new_position
            self.prey_positions[new_agent] = new_position
            self.agent_energies[new_agent] = self.initial_energy_prey
            self.grid_world_state[2, *new_position] = self.initial_energy_prey
            self.current_num_prey += 1
            occupied_positions.add(new_position)
            spawned_any = True

            # Not a birth: no reward, episode_births/reproduction_reward_prey untouched. Observation
            # is filled in later (see the `replenished` block above), not here -- setting it now would
            # still leave it stale once a LATER replacement in this same loop is spawned.
            rewards[new_agent] = 0.0
            self.cumulative_rewards[new_agent] = 0.0
            terminations[new_agent] = False
            truncations[new_agent] = False
            if self.verbose_spawning:
                print(f"FixedPreyDensityEnv: replacement prey {new_agent} spawned at {new_position}")

        self.agents.sort()  # match the base class's end-of-step invariant (predpreygrass_rllib_env.py Step 6)
        return spawned_any
