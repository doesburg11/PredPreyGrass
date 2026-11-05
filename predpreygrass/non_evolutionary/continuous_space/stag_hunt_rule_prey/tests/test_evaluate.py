"""Tests of the evaluation's joint-arrival count and chance level."""

import random

import numpy as np

from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import MOVES, StagHuntRulePrey
from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.evaluate import joint_arrivals, outcome, shifted


def test_joint_arrivals_on_tracks():
    # One mammoth standing at (200, 100); two predators arrive 2 decisions
    # apart (joint), leave, then one arrives alone (touched, not joint).
    mammoth = {"mammoth_0": {d: (200, 100) for d in range(10)}}
    far = (200, 600)
    a = {0: (200, 130), 1: (200, 130), 2: (200, 130), 3: far, 4: far, 5: (200, 130), 6: far}
    b = {0: far, 1: far, 2: (230, 100), 3: far, 4: far, 5: far, 6: far}
    result = joint_arrivals({"predator_0": a, "predator_1": b}, mammoth, list(range(7)), 800, 800, 48, 96)
    assert result == (2, 2, 1)


def test_shift_keeps_tracks_rigid():
    tracks = {"predator_0": {0: (10.0, 20.0), 1: (790.0, 30.0)}}
    moved = shifted(tracks, random.Random(1), 800, 800)["predator_0"]
    step = [(moved[1][i] - moved[0][i]) % 800 for i in range(2)]
    assert np.allclose(step, [780.0, 10.0])


def test_tracked_count_matches_the_environment():
    env = StagHuntRulePrey({"predator_count": 12, "mammoth_count": 6})
    predators, mammoths, decisions = {}, {}, []
    record = env._record

    def recording(joins):
        decisions.append(env.decision)
        for a in env.predators:
            predators.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        for a in env.mammoths:
            mammoths.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        record(joins)

    env._record = recording
    env.reset(seed=4)
    rng = np.random.default_rng(4)
    for _ in range(150):
        actions = {a: np.array([rng.integers(MOVES), rng.integers(2)]) for a in env.active}
        _, _, te, tr, _ = env.step(actions)
        if te["__all__"] or tr["__all__"]:
            break
    else:
        for mammoth_id in list(env.encounters):
            env._close_encounter(mammoth_id)
    stats = env.hunt_stats
    assert stats["encounter_episodes"] > 0
    assert joint_arrivals(predators, mammoths, decisions, 800, 800, 48, 96) == (
        stats["encounter_episodes"],
        stats["touched_episodes"],
        stats["joint_arrivals"],
    )


def test_outcome():
    env = StagHuntRulePrey()
    env.reset(seed=0)
    assert outcome(env) == "coexisting"
    env.animals["mammoth"].clear()
    assert outcome(env) == "only_predator_rabbit"


def test_tracked_count_with_births_kills_and_an_ending():
    """Births of both species, successful hunts and an episode that ends in
    a physics step: the tracks still give the environment's counts."""
    env = StagHuntRulePrey(
        {
            "predator_count": 12,
            "mammoth_count": 8,
            "hunt_energy_ratio": 0.05,
            "hunt_min_party": 1,
            "energy_predator_initial": 11.5,
            "energy_mammoth_initial": 23.5,
            "max_time_steps": 1600,
        }
    )
    predators, mammoths, decisions = {}, {}, []
    record = env._record

    def recording(joins):
        decisions.append(env.decision)
        for a in env.predators:
            predators.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        for a in env.mammoths:
            mammoths.setdefault(a.id, {})[env.decision] = (a.position.x, a.position.y)
        record(joins)

    env._record = recording
    env.reset(seed=7)
    rng = np.random.default_rng(7)
    while True:
        actions = {a: np.array([rng.integers(MOVES), 1]) for a in env.active}
        _, _, te, tr, _ = env.step(actions)
        if te["__all__"] or tr["__all__"]:
            break
    stats = env.episode_hunt_stats
    assert stats["births_predator"] > 0 and stats["births_mammoth"] > 0
    assert sum(stats[f"kills_party_{n}"] for n in range(1, 5)) > 0
    assert joint_arrivals(predators, mammoths, decisions, 800, 800, 48, 96) == (
        stats["encounter_episodes"],
        stats["touched_episodes"],
        stats["joint_arrivals"],
    )


def test_evaluation_plays_whole_episodes(monkeypatch):
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import checkpoints, evaluate
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import DEFAULT_CONFIG

    monkeypatch.setattr(checkpoints, "checkpoint_env_config", lambda checkpoint: dict(DEFAULT_CONFIG, truncate_on_prey_extinction=True))
    monkeypatch.setattr(checkpoints, "load_modules", lambda checkpoint: {})
    monkeypatch.setattr(evaluate, "_worker", {})
    evaluate._init("some/checkpoint")
    assert evaluate._worker["env"].truncate_on_prey_extinction is False


def test_summary_has_hunt_energy_means():
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey import evaluate
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import new_hunt_stats

    stats = new_hunt_stats()
    stats.update(hunts_party_2=2, hunted_party_energy_total=30.0, hunted_mammoth_energy_total=24.0)
    row = {"outcome": "coexisting", "hunt_stats": stats, "tracked": {}, "null": {}, "extinct_at": {}}
    summary = evaluate.summarise([row])
    assert summary["hunted_party_energy_mean"] == 15.0 and summary["hunted_mammoth_energy_mean"] == 12.0


def _saved_run(tmp_path, config):
    import json

    run = tmp_path / "run"
    (run / "source_code").mkdir(parents=True)
    (run / "checkpoint").mkdir()
    (run / "source_code" / "settings.json").write_text(json.dumps({"config_env": config}))
    return run / "checkpoint"


def test_checkpoints_from_before_alive_reward_still_load(tmp_path):
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.checkpoints import LATER_SETTINGS, checkpoint_env_config
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import DEFAULT_CONFIG

    old = {k: v for k, v in DEFAULT_CONFIG.items() if k not in LATER_SETTINGS}
    config = checkpoint_env_config(_saved_run(tmp_path, old))
    assert config["alive_reward"] == 0.0 and config["alive_reward_shared"] is False


def test_checkpoints_missing_any_other_setting_are_refused(tmp_path):
    import pytest

    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.checkpoints import checkpoint_env_config
    from predpreygrass.non_evolutionary.continuous_space.stag_hunt_rule_prey.environment import DEFAULT_CONFIG

    old = {k: v for k, v in DEFAULT_CONFIG.items() if k != "width"}
    with pytest.raises(ValueError):
        checkpoint_env_config(_saved_run(tmp_path, old))
