import copy

import numpy as np
import torch

from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning import analyze_compound_gift as acg
from predpreygrass.non_evolutionary.discrete_space.predator_compound_provisioning.config_env import config_env as BASE


class GiveWhenTagged:
    def _forward_inference(self, batch):
        obs = batch["obs"].numpy()
        logits = np.zeros((len(obs), 18), dtype=np.float32)
        for i, row in enumerate(obs):
            if row[5].sum() == 1:
                logits[i, 9:] = 5.0
        return {"action_dist_inputs": torch.as_tensor(logits)}


class NeverGive:
    def _forward_inference(self, batch):
        logits = torch.zeros((len(batch["obs"]), 18))
        logits[:, 9:] = -100.0
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
              "max_steps": 40, "compound_gift": "on"})
    return c


def modules(pred):
    return {"predator_male_policy": pred, "predator_female_policy": pred, "prey_policy": Prey()}


def test_gift_bit_prob_is_sum_of_the_nine_gift_variants():
    rows = acg.rollout(config(), modules(GiveWhenTagged()), 4, 3)
    assert np.nanmean([r["predator_male_gap"] for r in rows]) > 0.45
    assert np.nanmean([r["gift_events_per1000_pred_steps"] for r in rows]) > 0


def test_never_give_produces_no_transfer_or_gap():
    rows = acg.rollout(config(), modules(NeverGive()), 3, 3)
    assert np.isclose(np.nanmean([r["predator_male_gap"] for r in rows]), 0.0)
    assert np.isclose(np.nanmean([r["gift_events_per1000_pred_steps"] for r in rows]), 0.0)


def test_summary_handles_empty_conditioned_data():
    assert acg.summarize([np.nan], np.random.default_rng(0)) is None


def test_pair_outcomes_and_per_sex_gifts_are_reported_and_consistent():
    rows = acg.rollout(config(), modules(GiveWhenTagged()), 4, 3)
    for r in rows:
        for key in ("births_per_1000_pred_steps", "post_birth_near_frac", "mean_predators",
                    "predator_male_gifts_per1000_sex_steps", "predator_female_gifts_per1000_sex_steps"):
            assert key in r
        assert r["births_per_1000_pred_steps"] >= 0
    assert np.nanmean([r["births_per_1000_pred_steps"] for r in rows]) > 0


def test_inert_run_requests_but_never_transfers():
    c = config()
    c["compound_gift"] = "inert"
    rows = acg.rollout(c, modules(GiveWhenTagged()), 3, 3)
    assert np.nanmean([r["gift_bit_attempts_per1000_pred_steps"] for r in rows]) > 0
    assert np.isclose(np.nanmean([r["gift_events_per1000_pred_steps"] for r in rows]), 0.0)


def test_rejects_non_compound_run_config():
    c = config()
    c.pop("compound_gift")
    import pytest
    with pytest.raises(SystemExit):
        acg.rollout(c, modules(NeverGive()), 1, 3)
