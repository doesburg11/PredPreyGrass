"""
Exact-density variant of `FixedPreyDensityEnv` (Iteration 11), adding a predator-side analog for the
same purpose `predator_population_cap` was built for (RESULTS.md, Iteration 13): holding predator
population near a target across odds conditions, so a high-success run's behavior can be compared
against a low-success run's without population size confounding the comparison.

`predator_population_cap` does this by BLOCKING reproduction once population is already at or above
a ceiling (predpreygrass_rllib_env.py's Step 5b). Iteration 13's own Codex review flagged the real
cost of that design: blocking changes WHO gets to reproduce (whichever pairs are processed first in
a step, an arbitrary agent-ID sort-order tie-break), a reproductive-selection effect the uncapped
runs don't have -- so Iteration 13 could not cleanly separate "population size" from "the cap's own
selection effect" as the explanation for its findings.

This class removes that specific confound by never touching reproduction eligibility at all:

  - Reproduction happens exactly as in the base class. Every eligible, in-range pair succeeds, pays
    its normal birth cost, forms the normal mate bond, and earns the normal reproduction reward --
    nothing about WHO gets to reproduce is touched.
  - If total predator population (male + female) exceeds `predator_density_target` after a step,
    enough predators are removed to bring it back down to the target, chosen UNIFORMLY AT RANDOM
    among all currently alive predators of EITHER sex -- not by reproductive eligibility, fitness,
    hunting success, or any sort-order tie-break, so the cull is statistically independent of
    anything the trained policies are being evaluated on.
  - If population drops below the target, replacement predators are spawned (random sex, default
    starting energy, no parent or mate record), mirroring `FixedPreyDensityEnv`'s own prey-floor
    replenishment pattern (same `_find_available_spawn_position` helper, same per-episode ID-pool
    budget, `n_possible_predator_male`/`n_possible_predator_female`, shared with normal births).

This is NOT a confound-free design either -- it trades one confound for a different, smaller one:
an exogenous, policy-independent random death risk is itself a new intervention that uncapped runs
don't have. The claim is only that it is a cleaner control than blocking reproduction, since it never
distorts the reproduction-seeking incentive the policies are actually trained and evaluated on; it
should be described as such, not as removing all confounds.

Culled deaths are folded into the existing `episode_deaths["predator_male"/"predator_female"]`
counters (surfaced as `deaths_predator_male`/`deaths_predator_female` in TensorBoard), indistinguishable
there from ordinary starvation or combat deaths -- not broken out as a separate metric, for simplicity.

Predator extinction (either sex reaching 0, including as a direct result of the cull) still ends the
episode normally, via the same `terminations["__all__"]` formula the base class and
`FixedPreyDensityEnv` already use. Max-steps truncation is also untouched.
"""
import numbers

from predpreygrass.non_evolutionary.predator_sexual_reproduction.fixed_prey_density_env import FixedPreyDensityEnv


class FixedPredatorDensityEnv(FixedPreyDensityEnv):
    def __init__(self, config):
        super().__init__(config)
        self.predator_density_target = config.get("predator_density_target", None)
        if self.predator_density_target is not None:
            if isinstance(self.predator_density_target, bool) or not isinstance(
                self.predator_density_target, numbers.Integral
            ):
                raise ValueError(
                    f"predator_density_target must be an int (got {self.predator_density_target!r})"
                )
            self.predator_density_target = int(self.predator_density_target)
            if self.predator_density_target < 0:
                raise ValueError(
                    f"predator_density_target must be >= 0 (got {self.predator_density_target})"
                )
            # Codex review: predator_population_cap lives in the shared base class (Step 5b) and
            # would silently still block reproduction here too if both were set, contradicting this
            # class's whole "reproduction is never blocked" premise. Reject the combination outright
            # rather than let it silently compromise -- the two mechanisms serve the same purpose and
            # are not meant to be combined.
            if self.predator_population_cap is not None:
                raise ValueError(
                    "predator_density_target and predator_population_cap are mutually exclusive "
                    f"(got predator_density_target={self.predator_density_target}, "
                    f"predator_population_cap={self.predator_population_cap}) -- both hold predator "
                    "population near a target, and combining them would let the cap silently block "
                    "reproduction, defeating this class's purpose."
                )

    def step(self, action_dict):
        observations, rewards, terminations, truncations, infos = super().step(action_dict)

        # Max-steps truncation (the base class's early-return branch): leave untouched.
        if truncations.get("__all__", False):
            return observations, rewards, terminations, truncations, infos

        # Predator extinction ends the episode normally; nothing below can un-extinguish a sex, and
        # attempting to cull/replenish an already-extinct population isn't meaningful.
        if self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0:
            return observations, rewards, terminations, truncations, infos

        changed = False
        if self.predator_density_target is not None:
            total = self.current_num_predator_male + self.current_num_predator_female
            if total > self.predator_density_target:
                changed = self._cull_predators_to_target(observations, rewards, terminations, truncations)
            elif total < self.predator_density_target:
                changed = self._replenish_predators_to_target(observations, rewards, terminations, truncations)

        if changed:
            # Population changed for everyone (someone appeared or disappeared from the grid) --
            # regenerate every live agent's observation, matching FixedPreyDensityEnv's own pattern
            # for its prey-floor replenishment.
            self.agents.sort()
            for agent in self.agents:
                if agent in self.agent_positions:
                    observations[agent] = self._get_observation(agent)

        # Re-derive using the same formula the base class and FixedPreyDensityEnv use: correct
        # whether population just went up (replenish) or down (cull, possibly to extinction).
        terminations["__all__"] = (
            self.current_num_prey <= 0 or self.current_num_predator_male <= 0 or self.current_num_predator_female <= 0
        )
        return observations, rewards, terminations, truncations, infos

    def _cull_predators_to_target(self, observations, rewards, terminations, truncations):
        """Removes (current total - target) live predators, chosen uniformly at random across both
        sexes -- deliberately NOT weighted by anything the trained policies are evaluated on.
        Returns True (always removes at least one predator when called, by the caller's own
        total > target guard)."""
        total = self.current_num_predator_male + self.current_num_predator_female
        n_to_cull = total - self.predator_density_target
        alive_predators = [a for a in self.agent_positions if "predator_male" in a or "predator_female" in a]
        chosen = self.rng.choice(len(alive_predators), size=n_to_cull, replace=False)
        for i in chosen:
            agent = alive_predators[i]
            # Final observation before removal, mirroring every other death path in the base class
            # (e.g. _resolve_hunting_attempt's combat-death branch).
            observations[agent] = self._get_observation(agent)
            rewards[agent] = rewards.get(agent, 0.0)  # no extra penalty -- an exogenous removal, not a policy failure
            terminations[agent] = True
            truncations[agent] = False
            position = self.agent_positions[agent]
            if "predator_male" in agent:
                self.current_num_predator_male -= 1
                self.episode_deaths["predator_male"] += 1
            else:
                self.current_num_predator_female -= 1
                self.episode_deaths["predator_female"] += 1
            self.grid_world_state[1, *position] = 0
            del self.predator_positions[agent]
            del self.agent_positions[agent]
            del self.agent_energies[agent]
            # Not marked by the base class's own Step 4 (which already ran, inside super().step()),
            # so this environment must queue the removal itself for the standard cleanup at the top
            # of the NEXT step() call (mirrors every other death path's eventual self.agents.remove).
            self._pending_removal.append(agent)
            if self.verbose_spawning:
                print(f"FixedPredatorDensityEnv: culled {agent} at {position} to hold population near the target")
        return True

    def _replenish_predators_to_target(self, observations, rewards, terminations, truncations):
        """Spawns (target - current total) replacement predators, random sex, no parent/mate record
        -- mirroring FixedPreyDensityEnv's own prey-floor replenishment. Returns True iff at least
        one replacement was actually spawned (the caller uses this to decide whether every agent's
        observation needs to be regenerated)."""
        total = self.current_num_predator_male + self.current_num_predator_female
        deficit = self.predator_density_target - total
        occupied_positions = set(self.agent_positions.values())
        spawned_any = False
        for _ in range(deficit):
            male_has_room = self._next_predator_male_idx < self.n_possible_predator_male
            female_has_room = self._next_predator_female_idx < self.n_possible_predator_female
            if not male_has_room and not female_has_room:
                if self.verbose_spawning:
                    print("FixedPredatorDensityEnv: no new predator agent IDs left in either pool this episode")
                break
            # Codex review: choosing a sex uniformly first and only THEN checking its pool could abort
            # the whole loop on one exhausted-sex coin-flip even when the other sex still has plenty
            # of room. Choose only among sexes that still have room; fall back to whichever one does
            # when only one is available.
            if male_has_room and female_has_room:
                sex = "predator_male" if self.rng.random() < 0.5 else "predator_female"
            else:
                sex = "predator_male" if male_has_room else "predator_female"

            reference_position = (int(self.rng.integers(self.grid_size)), int(self.rng.integers(self.grid_size)))
            new_position = self._find_available_spawn_position(reference_position, occupied_positions)
            if new_position is None:
                if self.verbose_spawning:
                    print("FixedPredatorDensityEnv: no free spawn position available for a replacement predator")
                break

            if sex == "predator_male":
                new_agent = f"predator_male_{self._next_predator_male_idx}"
                self._next_predator_male_idx += 1
                offspring_energy = self.initial_energy_predator_male
                self.current_num_predator_male += 1
            else:
                new_agent = f"predator_female_{self._next_predator_female_idx}"
                self._next_predator_female_idx += 1
                offspring_energy = self.initial_energy_predator_female
                self.current_num_predator_female += 1

            self.agents.append(new_agent)
            self.agent_positions[new_agent] = new_position
            self.predator_positions[new_agent] = new_position
            self.agent_energies[new_agent] = offspring_energy
            self.grid_world_state[1, *new_position] = offspring_energy
            occupied_positions.add(new_position)
            spawned_any = True

            # Not a birth: no reward, episode_births untouched, no agent_parents/agent_mate entry
            # (a replacement has no parents and starts unpaired, same as an agent at reset). Real
            # observation is filled in later (the `changed` block in step()), not here -- setting it
            # now would still leave it stale once a LATER replacement in this same loop is spawned.
            rewards[new_agent] = 0.0
            self.cumulative_rewards[new_agent] = 0.0
            terminations[new_agent] = False
            truncations[new_agent] = False
            if self.verbose_spawning:
                print(f"FixedPredatorDensityEnv: replacement {new_agent} spawned at {new_position}")

        return spawned_any
