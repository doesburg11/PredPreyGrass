"""Trial 13 driver: wraps flagship's `PredPreyGrass.step()` with genome-driven
prey action selection (evolved eval_weights reward + REINFORCE-learned
action_weights) and a centrally-learning predator (see config.py's "Predator
strategy" note -- centralized_predator.CentralizedPredatorPolicy, one shared
policy updated from every predator's own experience), plus the newborn/
parent-detection and offspring-count bookkeeping flagship's env has no concept
of.

Structural analog of eco_evolutionary_erl_baldwin/world.py's `_step_agents`/
`_handle_agent_reproduction`, but wrapping flagship's actual env rather than
reimplementing grid mechanics -- see this module's README.md.
"""

from dataclasses import dataclass

import numpy as np

from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.centralized_predator import CentralizedPredatorPolicy
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.config import N_ACTIONS, OBS_DIM
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.features import extract_prey_features
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.genome import Genome, founder_genome, mutate
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.networks import (
    action_probs,
    evaluate,
    reinforce_update,
    sample_action,
)
from predpreygrass.evolutionary.eco_evolutionary_erl_flagship.predator_features import extract_predator_features
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass


@dataclass
class PredatorTrainingState:
    """Per-predator temporal bookkeeping for the ONE shared CentralizedPredatorPolicy
    -- no genome, no offspring_count: the learned policy itself is shared across
    every predator, not individually inherited, so a newborn predator just starts
    participating in the same policy immediately (registered lazily, see
    Trial13Driver._select_predator_action)."""

    agent_id: str
    prev_features: np.ndarray | None = None
    prev_action: int | None = None
    prev_energy: float | None = None


@dataclass
class PreyGenomeState:
    """Per-prey bookkeeping for one flagship `prey_<i>` agent_id -- structurally
    analogous to erl_baldwin's `Agent`, and exposing exactly the attributes
    metrics.py's `lineage_record`/`lineage_fieldnames` expect (agent_id,
    generation, born_step, offspring_count, genome)."""

    agent_id: str
    genome: Genome
    action_weights: np.ndarray  # LIVE, learned copy -- diverges from genome.action_weights over life
    action_bias: np.ndarray
    generation: int
    born_step: int
    prev_obs: np.ndarray | None = None
    prev_action: int | None = None
    prev_eval: float | None = None
    offspring_count: int = 0
    # A lineage TAG, not a genome trait -- inherited unchanged parent-to-child (never
    # mutated, unlike genome.eval_weights), see _handle_reproduction. When True,
    # eval_weights is ignored entirely: reinforcement comes directly from the
    # environment's own reproduction reward, not the genome's evaluation network.
    # Faithfully represents "+10 on reproduction, nothing else" -- a signal that can't
    # be expressed AS a genome, since reproduction isn't one of the 8 observed features
    # (see sparse_reward_check.py, README.md's status section).
    sparse_mode: bool = False


class Trial13Driver:
    """Owns the flagship env, the shared predator policy, and the prey genome
    registry; `step()` is the whole per-step loop. `on_agent_death(state,
    death_step)`, if set, is called for each prey death (mirrors erl_baldwin's
    World.on_agent_death hook, used for lineage-fitness CSV logging -- see
    run_trial13_simulation.py). No death hook for predators -- they aren't
    individually lineage-tracked, since their policy is shared, not evolved."""

    def __init__(self, env: PredPreyGrass, predator_policy: CentralizedPredatorPolicy, cfg: dict, rng: np.random.Generator):
        self.env = env
        self.predator_policy = predator_policy
        self.cfg = cfg
        self.rng = rng
        self.current_step = 0
        self.registry: dict[str, PreyGenomeState] = {}
        self.predator_registry: dict[str, PredatorTrainingState] = {}
        self._predators_reproduced_last_step: set[str] = set()
        self._prey_reproduced_last_step: set[str] = set()
        self.on_agent_death = None

    def reset(self):
        # Must pass seed= explicitly: PredPreyGrass.reset() only (re)seeds its
        # internal self.rng -- which drives founder agent/grass placement --
        # when given one. seed=None (the default if omitted) draws fresh OS
        # entropy instead (predpreygrass_rllib_env.py:141-142), which was
        # silently making every Trial 13 run's founder positions non-
        # reproducible regardless of --seed, found via two identical-seed CLI
        # runs producing wildly different population trajectories from the
        # very first logged step.
        self.env.reset(seed=self.cfg.get("seed"))
        self.current_step = 0
        self.registry = {}
        self.predator_registry = {}
        self._predators_reproduced_last_step = set()
        self._prey_reproduced_last_step = set()
        for agent_id in list(self.env.agents):
            if "prey" in agent_id:
                self.registry[agent_id] = self._new_founder(agent_id)
            elif "predator" in agent_id:
                self.predator_registry[agent_id] = PredatorTrainingState(agent_id=agent_id)

    def _new_founder(self, agent_id: str) -> PreyGenomeState:
        # mixed_founder_weights (a pair of eval_weights vectors), if set, randomly assigns
        # each founder to ONE of the two clusters (50/50) instead of a single shared
        # fixed_eval_weights or a fully random init -- for testing whether an
        # ALREADY-established two-strategy split is maintained by selection (as opposed to
        # fixed_eval_weights/polymorphism_check.py's neutral-start "does it emerge" question).
        # See polymorphism_maintenance_check.py.
        mixed = self.cfg.get("mixed_founder_weights")
        # sparse_reproduction_cluster ("a"/"b"), if set together with mixed_founder_weights,
        # marks whichever cluster it names as sparse_mode=True -- see PreyGenomeState.sparse_mode
        # and sparse_reward_check.py. sparse_reproduction_prey (a plain bool) does the same for a
        # single-cluster (non-mixed) founder population.
        sparse_cluster = self.cfg.get("sparse_reproduction_cluster")
        if mixed is not None:
            vec_a, vec_b = mixed
            if self.rng.random() < 0.5:
                fixed_eval_weights, sparse_mode = vec_a, sparse_cluster == "a"
            else:
                fixed_eval_weights, sparse_mode = vec_b, sparse_cluster == "b"
        else:
            fixed_eval_weights = self.cfg.get("fixed_eval_weights")
            sparse_mode = bool(self.cfg.get("sparse_reproduction_prey", False))
        genome = founder_genome(
            OBS_DIM, N_ACTIONS, self.rng, self.cfg["founder_weight_std"], fixed_eval_weights
        )
        return PreyGenomeState(
            agent_id=agent_id,
            genome=genome,
            action_weights=genome.action_weights.copy(),
            action_bias=genome.action_bias.copy(),
            generation=0,
            born_step=self.current_step,
            sparse_mode=sparse_mode,
        )

    def step(self):
        """One environment step: genome-driven prey actions + frozen-predator
        actions -> env.step() -> newborn/parent detection -> death detection."""
        action_dict = {}
        for agent_id in self.env.agents:
            if agent_id not in self.env.agent_positions:
                # Terminated in a PRIOR step but not yet purged from self.env.agents
                # -- flagship defers that removal to the top of its OWN next
                # step() call (predpreygrass_rllib_env.py:238-241), so it's still
                # in this list; skip it exactly as the env's own Step 3 does
                # ("if agent not in self.agent_positions: continue").
                continue
            if "prey" in agent_id:
                action_dict[agent_id] = self._select_prey_action(agent_id)
            elif "predator" in agent_id:
                action_dict[agent_id] = self._select_predator_action(agent_id)

        observations, rewards, terminations, truncations, infos = self.env.step(action_dict)
        self.current_step = self.env.current_step

        self._handle_reproduction(rewards)
        self._handle_deaths(terminations)
        # Consumed at the TOP of the NEXT step()'s _select_predator_action calls,
        # to correct the reinforcement credited to whatever action a predator
        # took on THIS step -- see that method's docstring for why.
        self._predators_reproduced_last_step = {
            agent_id for agent_id, r in rewards.items()
            if "predator" in agent_id and r == self.env.reproduction_reward_predator
        }
        # Same idea, for sparse_mode prey (see _select_prey_action): the env's OWN
        # reproduction signal, not anything derived from genome.eval_weights.
        self._prey_reproduced_last_step = {
            agent_id for agent_id, r in rewards.items()
            if "prey" in agent_id and r == self.env.reproduction_reward_prey
        }

        return observations, rewards, terminations, truncations, infos

    def _select_prey_action(self, agent_id: str) -> int:
        state = self.registry[agent_id]
        feat = extract_prey_features(self.env, agent_id)

        if state.sparse_mode:
            # Bypasses genome.eval_weights entirely -- faithfully represents "+10 on
            # reproduction, nothing else" (can't be expressed AS a genome: reproduction
            # isn't one of the 8 observed features). e_now/prev_eval stay unused;
            # reinforcement is exactly 0.0 on every non-reproduction step, which
            # reinforce_update's zero-reinforcement guard turns into a true no-op --
            # not an approximation of sparse reward, the actual thing.
            e_now = 0.0
            if state.prev_obs is not None:
                reinforcement = (
                    self.env.reproduction_reward_prey if agent_id in self._prey_reproduced_last_step else 0.0
                )
                reinforce_update(
                    state.action_weights, state.action_bias,
                    state.prev_obs, state.prev_action, reinforcement,
                    self.cfg["lr_positive"], self.cfg["lr_negative"],
                )
            probs = action_probs(feat, state.action_weights, state.action_bias)
            action = sample_action(probs, self.rng)
            state.prev_obs = feat
            state.prev_action = action
            state.prev_eval = e_now
            return action

        e_now = evaluate(feat, state.genome.eval_weights, state.genome.eval_bias)

        if state.prev_obs is not None:
            reinforcement = e_now - state.prev_eval
            reinforce_update(
                state.action_weights, state.action_bias,
                state.prev_obs, state.prev_action, reinforcement,
                self.cfg["lr_positive"], self.cfg["lr_negative"],
            )

        probs = action_probs(feat, state.action_weights, state.action_bias)
        action = sample_action(probs, self.rng)

        state.prev_obs = feat
        state.prev_action = action
        state.prev_eval = e_now
        return action

    def _select_predator_action(self, agent_id: str) -> int:
        """Unlike prey (an evolved, per-agent intrinsic reward), the predator's
        reward is the real, directly observable net energy change from its
        PREVIOUS step's action, relative to the BASELINE of doing nothing --
        i.e. energy_change + energy_loss_per_step_predator, not raw energy
        change. This subtraction matters: raw energy change is negative on
        almost every step regardless of action (the ambient per-step drain,
        -0.15 by default, dwarfs the rare +catch spikes), so training on it
        directly mostly teaches "whatever I just did was bad" uniformly rather
        than "catching is good" -- confirmed directly: predator_action_weight_
        absmean barely moved (0.4037 -> 0.4027) across a 400-step run before
        going extinct, consistent with a systematically-biased, uninformative
        signal rather than real learning. Baseline-subtracting makes an
        ordinary no-catch step net to ~0 reinforcement (a no-op, see
        CentralizedPredatorPolicy.update/networks.reinforce_update's zero-
        reinforcement guard) and a catch a clear, isolated positive spike --
        the actual informative signal, not buried in ambient noise.

        A second baseline correction: reproduction ALSO costs the parent
        `initial_energy_predator` energy on top of the ambient drain
        (predpreygrass_rllib_env.py:423), and that cost lands on the SAME step
        as whatever action the predator happened to take -- without
        correction, reproducing (a good outcome, reflecting past hunting
        success) would show up as a large spurious NEGATIVE reinforcement for
        an essentially arbitrary action, since reproduction is triggered by
        accumulated energy crossing a threshold, not caused by that step's
        action. `_predators_reproduced_last_step` (set in step(), from the
        env's own reproduction_reward_predator signal) corrects for this too.

        Every predator's transition updates the SAME shared policy (see
        CentralizedPredatorPolicy.update), so learning pools across the whole
        predator population rather than each having to rediscover hunting
        independently. New predator ids (newborns) are registered lazily here
        rather than in _handle_reproduction -- no genome/parent-pairing is
        needed since there's nothing per-agent to inherit, just the shared
        policy every predator already uses."""
        if agent_id not in self.predator_registry:
            self.predator_registry[agent_id] = PredatorTrainingState(agent_id=agent_id)
        state = self.predator_registry[agent_id]

        feat = extract_predator_features(self.env, agent_id)
        current_energy = self.env.agent_energies[agent_id]

        if state.prev_features is not None:
            reinforcement = (
                current_energy - state.prev_energy + self.env.energy_loss_per_step_predator
            )
            if agent_id in self._predators_reproduced_last_step:
                reinforcement += self.env.initial_energy_predator
            self.predator_policy.update(state.prev_features, state.prev_action, reinforcement)

        action = self.predator_policy.act(feat, self.rng)

        state.prev_features = feat
        state.prev_action = action
        state.prev_energy = current_energy
        return action

    def _handle_reproduction(self, rewards: dict):
        """A prey agent whose reward this step equals reproduction_reward_prey just
        reproduced (predpreygrass_rllib_env.py:441-469); flagship places the
        newborn adjacent to it, with a fresh id already present in self.env.agents
        by the time env.step() returns. Detected here from the outside -- no
        modification to flagship's env -- by pairing each such parent to the
        nearest not-yet-registered prey id (the env always spawns the child
        adjacent to its parent, see _find_available_spawn_position)."""
        if not rewards:
            return
        threshold = self.env.reproduction_reward_prey
        parents = [
            agent_id for agent_id, r in rewards.items()
            if "prey" in agent_id and r == threshold and agent_id in self.registry
        ]
        if not parents:
            return
        unmatched_newborns = [
            agent_id for agent_id in self.env.agents
            if "prey" in agent_id and agent_id not in self.registry
        ]
        for parent_id in parents:
            if not unmatched_newborns:
                break  # more reproduction rewards than detected newborns; shouldn't happen
            parent_pos = self.env.agent_positions[parent_id]
            best_id, best_dist = None, None
            for newborn_id in unmatched_newborns:
                pos = self.env.agent_positions[newborn_id]
                dist = abs(pos[0] - parent_pos[0]) + abs(pos[1] - parent_pos[1])
                if best_dist is None or dist < best_dist:
                    best_id, best_dist = newborn_id, dist
            unmatched_newborns.remove(best_id)

            parent_state = self.registry[parent_id]
            parent_state.offspring_count += 1
            child_genome = mutate(
                parent_state.genome, self.rng, self.cfg["mutation_rate"], self.cfg["mutation_std"]
            )
            self.registry[best_id] = PreyGenomeState(
                agent_id=best_id,
                genome=child_genome,
                action_weights=child_genome.action_weights.copy(),
                action_bias=child_genome.action_bias.copy(),
                generation=parent_state.generation + 1,
                born_step=self.current_step,
                sparse_mode=parent_state.sparse_mode,  # a lineage tag, inherited exactly -- never mutated
            )

    def _handle_deaths(self, terminations: dict):
        for agent_id, terminated in terminations.items():
            if not terminated:
                continue
            if "prey" in agent_id and agent_id in self.registry:
                state = self.registry.pop(agent_id)
                if self.on_agent_death is not None:
                    self.on_agent_death(state, self.current_step)
            elif "predator" in agent_id:
                self.predator_registry.pop(agent_id, None)

    def population_counts(self) -> dict[str, int]:
        return {
            "prey": sum(1 for a in self.env.agents if "prey" in a and a in self.env.agent_positions),
            "predator": sum(1 for a in self.env.agents if "predator" in a and a in self.env.agent_positions),
        }

    def genome_stats(self) -> dict[str, float]:
        if not self.registry:
            stats = {"eval_weight_absmean": float("nan"), "action_weight_absmean": float("nan")}
        else:
            eval_abs = np.concatenate([np.abs(s.genome.eval_weights) for s in self.registry.values()])
            action_abs = np.concatenate([np.abs(s.action_weights).ravel() for s in self.registry.values()])
            stats = {
                "eval_weight_absmean": float(eval_abs.mean()),
                "action_weight_absmean": float(action_abs.mean()),
            }
        stats["predator_action_weight_absmean"] = self.predator_policy.action_weight_absmean()
        return stats
