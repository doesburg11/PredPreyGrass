"""Replayed deer (deer_schedule.py, RESULTS.md section 2): the schedule D(t),
copying with a manifest, source checks, which deer a checkpoint acts with,
and arm E's per-environment switching (evolve.py)."""

import json
import pickle

import numpy as np
import pytest

from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import deer_schedule as ds
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt import evolve
from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.config.config_env import config_env


@pytest.mark.parametrize("t, d", [(1, 10), (9, 10), (10, 10), (19, 10), (20, 20), (21, 20), (29, 20), (30, 30), (990, 990), (999, 990), (1000, 1000)])
def test_schedule(t, d):
    assert ds.source_iteration(t) == d


@pytest.mark.parametrize("t", [0, 1001, 1.0, True])
def test_schedule_refuses_other_iterations(t):
    with pytest.raises(ValueError):
        ds.source_iteration(t)


def test_switches_and_sources():
    assert ds.switch_iterations(1000) == list(range(20, 1001, 10))
    assert ds.switch_iterations(19) == []
    assert ds.switch_iterations(21) == [20]
    assert ds.source_iterations(1000) == list(range(10, 1001, 10))
    assert ds.source_iterations(500) == list(range(10, 501, 10))


def fake_source(root, name="src", iterations=(10, 20, 30), env=None, rule="touch", trained=1000, extra=None):
    """A deer_stag_hunt-like run with deer modules of distinct weights."""
    run = root / name
    (run / "source_code").mkdir(parents=True)
    source_env = {k: v for k, v in (env or config_env).items() if not k.startswith("gene_")}
    source_env["deer_catch_rule"] = rule
    source_env.update(extra or {})
    (run / "source_code" / "settings.json").write_text(json.dumps({"config_env": source_env, "config_ppo": {"iterations": trained, "vf_share_layers": False}}))
    for it in iterations:
        module = ds.module_dir(ds.source_checkpoint(run, it))
        module.mkdir(parents=True)
        with open(module / "module_state.pkl", "wb") as f:
            pickle.dump({"pi.w": np.full(3, float(it))}, f)
        (module / "metadata.json").write_text("{}")
    return run


def test_check_source_accepts_the_same_world_and_old_settings_without_new_keys(tmp_path):
    env = {k: v for k, v in config_env.items() if k != "deer_young_solo"}
    ds.check_source(fake_source(tmp_path, env=env), config_env)


@pytest.mark.parametrize("kwargs", [
    {"rule": "party"},
    {"trained": 500},
    {"extra": {"deer_young_solo": "certain"}},
    {"extra": {"grass_regrowth": "logistic"}},
    {"extra": {"energy_grass_gain": 3.0}},
])
def test_check_source_refuses_another_world(tmp_path, kwargs):
    with pytest.raises(ValueError):
        ds.check_source(fake_source(tmp_path, **kwargs), config_env)


def test_copy_schedule_manifest_hashes_and_tampering(tmp_path):
    source = fake_source(tmp_path)
    run = tmp_path / "run"
    manifest = ds.copy_schedule(source, run, 30, config_env)
    assert sorted(manifest["deer"]) == ["10", "20", "30"]
    for it in (10, 20, 30):
        assert ds.load_state(ds.deer_path(run, it))["pi.w"][0] == it
    with pytest.raises(ValueError):
        ds.copy_schedule(source, run, 30, config_env)  # refuses to overwrite
    with pytest.raises(ValueError):
        ds.deer_path(run, 40)
    with open(run / ds.SCHEDULE / "iter_000020" / "module_state.pkl", "wb") as f:
        pickle.dump({"pi.w": np.zeros(3)}, f)
    with pytest.raises(RuntimeError):
        ds.deer_path(run, 20)


def test_copy_schedule_refuses_a_missing_checkpoint(tmp_path):
    source = fake_source(tmp_path, iterations=(10, 30))
    with pytest.raises(ValueError):
        ds.copy_schedule(source, tmp_path / "run", 30, config_env)


def test_checkpoint_deer_is_the_recorded_one(tmp_path):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import checkpoint_deer

    source = fake_source(tmp_path)
    run = tmp_path / "run"
    (run / "source_code").mkdir(parents=True)
    (run / "source_code" / "settings.json").write_text(json.dumps({"config_env": {}, "config_ppo": {}}))
    ds.copy_schedule(source, run, 30, config_env)
    checkpoint = run / "checkpoint" / "iter_000020"
    ds.write_active(checkpoint, 20, run)
    assert checkpoint_deer(checkpoint) == run / ds.SCHEDULE / "iter_000020"
    with pytest.raises(ValueError):
        checkpoint_deer(run / "checkpoint" / "iter_000030")  # replaying run, nothing recorded
    (checkpoint / ds.ACTIVE).write_text(json.dumps({"source_iteration": 20, "sha256": "0" * 64}))
    with pytest.raises(RuntimeError):
        checkpoint_deer(checkpoint)


def test_section_1_checkpoints_keep_their_frozen_deer(tmp_path):
    from predpreygrass.evolutionary.continuous_space.eco_evolutionary_deer_stag_hunt.checkpoints import DEER_POLICY, checkpoint_deer

    run = tmp_path / "old"
    (run / "source_code").mkdir(parents=True)
    (run / "source_code" / "settings.json").write_text("{}")
    (run / "checkpoint").mkdir()
    assert checkpoint_deer(run / "checkpoint") == run / DEER_POLICY


def iterations_log(run, steps):
    run.mkdir(parents=True)
    (run / "iterations.jsonl").write_text("".join(json.dumps({"iteration": i + 1, "physics_steps": s}) + "\n" for i, s in enumerate(steps)))


def test_e_switch_thresholds_are_the_median_at_the_end_of_the_iteration_before(tmp_path):
    # RESULTS.md section 2.1 clarification: when L and B switch, before sampling t
    for name, scale in (("a", 100), ("b", 200), ("c", 300)):
        iterations_log(tmp_path / name, [scale * (i + 1) for i in range(30)])
    thresholds = evolve.switch_thresholds([tmp_path / n for n in "abc"], 30)
    assert thresholds == {20: 200 * 19, 30: 200 * 29}


def test_e_switch_thresholds_need_a_complete_log(tmp_path):
    run = tmp_path / "gap"
    run.mkdir()
    (run / "iterations.jsonl").write_text(json.dumps({"iteration": 1, "physics_steps": 5}) + "\n" + json.dumps({"iteration": 3, "physics_steps": 9}) + "\n")
    with pytest.raises(SystemExit):
        evolve.switch_thresholds([run], 30)


def test_e_deer_clock_and_several_thresholds_in_one_step():
    switches = {20: 100, 30: 150, 40: 150, 50: 400}
    assert evolve.deer_at(switches, 0) == 10
    assert evolve.deer_at(switches, 99) == 10
    assert evolve.deer_at(switches, 100) == 20
    assert evolve.deer_at(switches, 160) == 40
    clock = evolve.DeerClock(switches)
    seen = [clock.active(s) for s in (0, 99, 100, 149, 151, 399, 400, 10_000)]
    assert seen == [10, 10, 20, 20, 40, 40, 50, 50]
    clock = evolve.DeerClock(switches)
    assert clock.active(500) == 50  # every threshold crossed at once
    with pytest.raises(ValueError):
        evolve.DeerClock({20: 200, 30: 100})


def test_e_switches_split_like_the_marks_so_a_mark_meets_its_deer():
    # thresholds at the end of iteration t - 1 are below the mark's total at t,
    # and the per-environment split keeps that order in every environment
    switches = evolve.worker_targets({250: 10_000_000 - 7}, 14)
    marks = evolve.worker_targets({250: 10_000_000}, 14)
    for env_switch, env_mark in zip(switches, marks):
        clock = evolve.DeerClock(env_switch)
        assert clock.active(env_mark[250]) == ds.source_iteration(250)
