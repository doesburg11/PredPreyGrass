"""measure_swarms end to end with stand-in policies (no checkpoint)."""

import math

import numpy as np
import pytest
import torch
from ray.rllib.core.columns import Columns

from predpreygrass.non_evolutionary.continuous_space.foraging_swarms import measure_swarms as ms
from predpreygrass.non_evolutionary.continuous_space.foraging_swarms.environment import (
    DIRECTIONS,
    FIXES,
    SPEEDS,
    PredPreyGrassContinuous,
)

WORLD = dict(FIXES, predator_count=8, prey_count=30, max_time_steps=600)


class Uniform:
    """Every action equally likely."""

    def forward_inference(self, batch):
        obs = batch[Columns.OBS]
        return {Columns.ACTION_DIST_INPUTS: torch.zeros(len(obs), DIRECTIONS * len(SPEEDS))}


class Follower:
    """Moves at full speed in the direction of its first conspecific's
    velocity (sharply peaked), a policy that follows its neighbours."""

    def __init__(self, species, lay):
        self.v = lay[species] + 4

    def forward_inference(self, batch):
        obs = batch[Columns.OBS]
        angles = torch.arange(DIRECTIONS, dtype=torch.float32) * (2 * math.pi / DIRECTIONS)
        units = torch.stack([torch.sin(angles), -torch.cos(angles)], dim=1)  # direction_vector
        score = obs[:, self.v : self.v + 2] @ units.T  # (n, 16)
        logits = torch.full((len(obs), DIRECTIONS * len(SPEEDS)), -50.0)
        logits[:, :DIRECTIONS] = 20 * score
        return {Columns.ACTION_DIST_INPUTS: logits}


def test_play_and_summarize_run_and_are_consistent():
    env = PredPreyGrassContinuous(dict(WORLD))
    lay = ms.layout(env)
    modules = {"predator": Uniform(), "prey": Uniform()}
    episodes = [ms.play(env, modules, seed, np.random.default_rng(seed)) for seed in (0, 1)]
    summary = ms.summarize(episodes, modules, lay)
    assert summary["episodes"] == 2
    for e in episodes:
        assert e["prey_caught"] >= 0 and e["prey_exposure"] > 0
        for species in ("predator", "prey"):
            for event in e["losses"][species]:
                assert 1 <= len(event["distances"]) <= ms.LOSS_WINDOW + 1
                assert event["reacquired"] is None or 1 <= event["reacquired"] <= ms.LOSS_WINDOW
    m = summary["means"]
    assert 0 < m["prey/clark_evans_R"] < 2
    assert -1 <= m["prey/alignment"] <= 1
    assert "prey/alignment_minus_shuffled" in summary["intervals"]
    i = summary["intervals"]["prey/clark_evans_R"]
    assert i["low"] <= i["estimate"] <= i["high"]
    ms.format_summary("test", summary)  # formats without errors


def test_probe_measures_a_follower_turning_with_its_neighbours():
    env = PredPreyGrassContinuous(dict(WORLD))
    lay = ms.layout(env)
    rng = np.random.default_rng(0)
    batch = np.zeros((200, env.obs_size), dtype=np.float32)
    start = lay["prey"]
    batch[:, start] = 1.0
    angles = rng.uniform(0, 2 * math.pi, 200)
    batch[:, start + 4], batch[:, start + 5] = 0.8 * np.cos(angles), 0.8 * np.sin(angles)
    out = ms.run_probe(Follower("prey", lay), batch, "prey", lay)
    for degrees in ("90", "180", "270"):
        turn = (out[degrees]["mean_turn_degrees"] - float(degrees) + 180) % 360 - 180
        assert abs(turn) < 15, (degrees, out[degrees])
        assert out[degrees]["tv_distance"] > 0.9
        assert out[degrees]["concentrated_share"] > 0.9
    ignorer = ms.run_probe(Uniform(), batch, "prey", lay)
    assert ignorer["90"]["tv_distance"] == pytest.approx(0.0)
    assert ignorer["90"]["concentrated_share"] == 0.0  # no clear direction: no turn measured


def test_probe_observations_need_a_moving_conspecific():
    env = PredPreyGrassContinuous(dict(WORLD, prey_observe_neighbor_movement=False))
    lay = ms.layout(env)
    episodes = [ms.play(env, {"predator": Uniform(), "prey": Uniform()}, 0, np.random.default_rng(0))]
    assert len(episodes[0]["probe"]["prey"]) == 0  # condition B: nothing to rotate
