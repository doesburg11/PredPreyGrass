import copy

import numpy as np
import torch

from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction import analyze_learned_gift as alg
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.config_env import config_env as BASE
from predpreygrass.non_evolutionary.discrete_space.predator_sexual_reproduction.predpreygrass_rllib_env import PredPreyGrass


class GiftWhenMateTagged:
    """Synthetic policy: choose give iff the sixth mate channel has a mark."""
    def _forward_inference(self, batch):
        obs = batch["obs"].numpy()
        logits = np.zeros((len(obs), 10), dtype=np.float32)
        for i, row in enumerate(obs):
            if row.shape[0] > 5 and row[5].sum() == 1:
                logits[i, 9] = 8.0
        return {"action_dist_inputs": torch.as_tensor(logits)}


class NeverGift:
    def _forward_inference(self, batch):
        logits = torch.zeros((len(batch["obs"]), 10))
        logits[:, 9] = -100.0
        return {"action_dist_inputs": logits}


class Prey:
    def _forward_inference(self, batch):
        return {"action_dist_inputs": torch.zeros((len(batch["obs"]), 9))}


def config():
    c = copy.deepcopy(BASE)
    c.update({"grid_size": 12, "initial_num_grass": 15, "initial_num_fruit": 15,
              "n_initial_active_predator_male": 4, "n_initial_active_predator_female": 4,
              "n_initial_active_prey": 4, "initial_energy_predator_male": 8.0,
              "initial_energy_predator_female": 8.0, "predator_creation_energy_threshold": 9.0,
              "max_steps": 40, "mate_obs_channel": True, "learned_gift": "on",
              "male_gift_donation_rate": 0.0})
    return c


def modules(pred):
    return {"predator_male_policy": pred, "predator_female_policy": pred, "prey_policy": Prey()}


def test_eligible_requires_all_physical_conditions():
    env = PredPreyGrass(config())
    env.reset(seed=1)
    male = next(a for a in env.agents if "predator_male" in a)
    female = next(a for a in env.agents if "predator_female" in a)
    env.agent_mate[male] = female
    env.agent_mate[female] = male
    env.predator_positions[male] = env.agent_positions[male] = (4, 4)
    env.predator_positions[female] = env.agent_positions[female] = (4, 5)
    env.agent_energies[male] = 7.0
    env.agent_energies[female] = 7.0
    assert alg.eligible(env, male)
    env.agent_energies[female] = env.predator_creation_energy_threshold
    assert not alg.eligible(env, male)
    env.agent_energies[female] = 7.0
    env.agent_energies[male] = 2.0
    assert not alg.eligible(env, male)


def test_rollout_detects_gift_specialization():
    rows = alg.rollout(config(), modules(GiftWhenMateTagged()), episodes=4, seed0=3)
    assert len(rows) == 4
    # Some states are eligible due to the deliberately low reproduction threshold.
    assert sum(r["predator_male_eligible_states"] for r in rows) > 0
    assert np.nanmean([r["predator_male_gap"] for r in rows]) > 0.5
    assert np.nanmean([r["gift_events_per1000_pred_steps"] for r in rows]) > 0


def test_never_gift_has_no_conditional_gap_or_transfers():
    rows = alg.rollout(config(), modules(NeverGift()), episodes=3, seed0=3)
    assert np.isclose(np.nanmean([r["predator_male_gap"] for r in rows]), 0.0)
    assert np.isclose(np.nanmean([r["gift_events_per1000_pred_steps"] for r in rows]), 0.0)


def test_summary_omits_undefined_episodes():
    rng = np.random.default_rng(0)
    assert alg.summarize([np.nan, 0.1, 0.3], rng)[0] == 0.2
    assert alg.summarize([np.nan], rng) is None
