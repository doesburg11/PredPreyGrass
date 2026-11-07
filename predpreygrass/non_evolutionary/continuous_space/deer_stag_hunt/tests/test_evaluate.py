"""Tests of evaluation and the fair cooperation measure with untrained
policies on short episodes."""

import pytest

from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt import evaluate
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.checkpoints import untrained_modules
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.environment import DEFAULT_CONFIG, KINDS, DeerStagHunt
from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.launchers import coop_fair


def short_world():
    config = dict(DEFAULT_CONFIG, max_time_steps=400)
    return DeerStagHunt(config), untrained_modules(config, seed=0)


def test_play_and_summary():
    env, modules = short_world()
    rows = [evaluate.play(env, modules, seed) for seed in (0, 1)]
    for row in rows:
        assert row["physics_steps"] <= 400 and row["stats"]["physics_steps"] == row["physics_steps"]
    again = evaluate.play(env, modules, 0)
    assert again == rows[0]  # seeded: the same episode
    summary = evaluate.summarise(rows)
    assert summary["episodes"] == 2 and sum(summary["outcomes"].values()) == 2
    assert summary["coexisting"] == sum(row["outcome"] == "coexisting" for row in rows)
    assert summary["totals"]["deer_caught"] == sum(row["stats"]["deer_caught"] for row in rows)


def test_coop_fair_counts_and_leaves_the_episode_unchanged():
    env, modules = short_world()
    counts = coop_fair.count(env, modules, 0)
    assert counts["hunter_steps"] > 0 and 0 <= counts["with_partner"] <= counts["hunter_steps"]
    for kind in KINDS:
        assert 0 <= counts[f"{kind}_with_partner"] <= counts[f"{kind}_approaches"]
    assert env._physics_step.__func__ is DeerStagHunt._physics_step and env._contacts.__func__ is DeerStagHunt._contacts
    assert evaluate.play(env, modules, 0) == evaluate.play(*short_world(), 0)


def test_coop_fair_summary():
    rows = [
        {"hunter_steps": 100, "with_partner": 10, "fawn_approaches": 10, "fawn_with_partner": 0, "middle_approaches": 0, "middle_with_partner": 0, "adult_approaches": 10, "adult_with_partner": 5},
        {"hunter_steps": 100, "with_partner": 30, "fawn_approaches": 0, "fawn_with_partner": 0, "middle_approaches": 5, "middle_with_partner": 1, "adult_approaches": 0, "adult_with_partner": 0},
    ]
    s = coop_fair.summarise(rows)
    assert s["chance"] == 0.2 and s["share"] == {"fawn": 0.0, "middle": 0.2, "adult": 0.5}
    assert s["adult_above_fawn_episodes"] == 1 and s["cooperation_shown"] is True


def test_coop_fair_bootstrap_and_ratio():
    rows = [
        {"hunter_steps": 100, "with_partner": 10, "fawn_approaches": 10, "fawn_with_partner": 0, "middle_approaches": 0, "middle_with_partner": 0, "adult_approaches": 10, "adult_with_partner": 5}
    ] * 10
    s = coop_fair.summarise(rows)
    assert s["adult_over_chance"] == 5.0 and s["adult_minus_chance"] == 0.4
    assert s["adult_minus_chance_interval"] == [0.4, 0.4] and s["adult_above_chance_episodes"] == 10
    empty = [dict(rows[0], adult_approaches=0, adult_with_partner=0)] * 10
    assert coop_fair.summarise(empty)["adult_minus_chance_interval"] is None


def test_verdict():
    from predpreygrass.non_evolutionary.continuous_space.deer_stag_hunt.launchers import verdict

    evaluation = {"episodes": 50, "coexisting": 45}
    coop = {"episodes": 20, "chance": 0.10, "share": {"fawn": 0.08, "middle": 0.09, "adult": 0.0905}}
    assert verdict.tests(evaluation, coop) == {"1_coexistence": True, "2_cooperation": False, "3_above_section_7": True}
    good = verdict.tests(evaluation, dict(coop, share={"fawn": 0.08, "middle": 0.09, "adult": 0.12}))
    assert all(good.values())
    assert not verdict.tests({"episodes": 50, "coexisting": 44}, coop)["1_coexistence"]
    bad = {"1_coexistence": True, "2_cooperation": False, "3_above_section_7": True}
    assert verdict.verdict({"run_1": good, "run_2": good, "run_3": bad})
    assert not verdict.verdict({"run_1": good, "run_2": bad, "run_3": bad})
    with pytest.raises(ValueError):
        verdict.verdict({"run_1": good, "run_2": good})
    with pytest.raises(ValueError):
        verdict.tests({"episodes": 20, "coexisting": 20}, coop)
