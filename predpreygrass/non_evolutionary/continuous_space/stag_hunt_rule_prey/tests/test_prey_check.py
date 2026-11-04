"""Tests of the prey check."""

import argparse

import numpy as np
import pytest

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import MOVES, StagHuntRulePrey
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.geometry import Vector
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.prey_check import IDLE_HUNTER, STRETCH, harmless_world, parse_setting, play_episode, summarise


def test_the_idle_hunter_lasts_the_episode_and_has_no_young():
    env = StagHuntRulePrey(dict(IDLE_HUNTER, max_time_steps=2 * STRETCH + 8))
    row = play_episode(env, 0)
    assert env.time_step == env.max_time_steps
    assert len(env.predators) == 1 and row["births"]["predator"] == 0
    assert set(row["stretches"]) == {"0", "1", "2"}


def test_the_hunter_catches_and_frightens_nothing():
    env = harmless_world(dict(IDLE_HUNTER, rabbit_count=1, grass_count=0, steppe_count=0))
    env.reset(seed=0)
    hunter, rabbit = env.predators[0], env.rabbits[0]
    rabbit.position = Vector(hunter.position.x + 5, hunter.position.y)
    rabbit.orientation = 0
    env.step({"predator_0": np.array([MOVES - 1, 0])})
    assert env.rabbits and env.hunt_stats["rabbits_caught"] == 0


def test_the_summary_counts_extinctions_and_means():
    rows = [
        {"seed": 0, "extinct_at": {"rabbit": 5}, "births": {"rabbit": 2, "mammoth": 0}, "stretches": {"0": [2.0, 4.0, 6.0, 1.0, 1.0]}},
        {"seed": 1, "extinct_at": {}, "births": {"rabbit": 4, "mammoth": 2}, "stretches": {"0": [4.0, 6.0, 8.0, 3.0, 3.0], "1": [1.0, 1.0, 1.0, 1.0, 1.0]}},
    ]
    summary = summarise(rows)
    assert summary["rabbits_out"] == 1 and summary["mammoths_out"] == 0
    assert summary["births"] == {"rabbit": 3.0, "mammoth": 1.0}
    assert summary["by_stretch"]["0"] == [3.0, 5.0, 7.0, 2.0, 2.0]
    assert summary["by_stretch"]["1"] == [1.0, 1.0, 1.0, 1.0, 1.0]


def test_settings_are_parsed():
    assert parse_setting("grass_respawn_delay=800") == ("grass_respawn_delay", 800)
    assert parse_setting("hunt_energy_ratio=0.5") == ("hunt_energy_ratio", 0.5)
    with pytest.raises(argparse.ArgumentTypeError):
        parse_setting("grass_respawn_delay")
